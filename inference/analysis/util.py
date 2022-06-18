import io
import json
import pathlib
import itertools
import copy
import sys
import pandas as pd
import numpy as np
import collections
import shutil
from pprint import pprint


modelRenames = {"complexCutlassGemm": "cGEMM"}

baselineName = "eTask"
kaasName = "kTask"


def updateThroughputRes(resDict):
    newDict = resDict[0]

    newDict['metrics_warm'] = newDict['metrics']
    del newDict['metrics']

    for metric in list(newDict['metrics_warm'].keys()):
        origValue = newDict['metrics_warm'][metric]
        newDict['metrics_warm'][metric] = {'mean': origValue}

    return newDict


def updateMlPerfRes(resDict):
    resDict = resDict[0]
    newDict = {}

    newDict['config'] = copy.deepcopy(resDict['config'])
    newDict['metrics_warm'] = copy.deepcopy(resDict['metrics'])

    for mName, mVal in resDict['metrics'].items():
        newDict['metrics_warm'][mName] = {"mean": mVal}

    newDict['metrics_warm']['t_response'] = {}
    newDict['metrics_warm']['t_response']['events'] = resDict['metrics']['latencies']
    return newDict


def updateFormat(suitePath, outPath, suiteType):
    if not outPath.exists():
        outPath.mkdir(0o700, parents=True)

    for expDir in suitePath.glob("*"):
        outExpDir = outPath / expDir.name
        outExpDir.mkdir(0o700)
        for resFile in expDir.glob("*_results.json"):
            with open(resFile, 'r') as f:
                resDict = json.load(f)

            if isinstance(resDict, dict):
                shutil.copy(resFile, outExpDir / resFile.name)
            else:
                if suiteType == 'throughput':
                    newDict = updateThroughputRes(resDict)
                elif suiteType == 'mlperf':
                    newDict = updateMlPerfRes(resDict)
                else:
                    raise RuntimeError("I don't know how to convert that yet")

                with open(outExpDir / resFile.name, 'w') as f:
                    json.dump(newDict, f)


def aggregateModels(fullResults, metric):
    """Take the output of loadAllMlperf and return a dictionary of dataframes
    indexed by model name. Each dataframe will be indexed by nReplicas, the
    columns are the modes, and the values are 'metric'

    Returns:
        {modelName: pd.DataFrame()}.

    """
    models = ['resnet50', 'bert', 'jacobi', 'complexCutlassGemm']
    resDfs = {}
    for model in models:
        modelRes = [res for res in fullResults.values() if model in res['config']['model']]
        if len(modelRes) == 0:
            continue
        maxNReplica = max([d['config']['n_replica'] for d in modelRes])

        fullIndex = list(range(1, maxNReplica+1))
        df = pd.DataFrame(index=fullIndex)

        tvmRes = [res for res in modelRes if res['config']['model_type'] in ['tvm', 'direct']]
        tvmSer = pd.Series([res[metric] for res in tvmRes], index=[res['config']['n_replica'] for res in tvmRes])
        # tvmSer = pd.Series([res['config']['t_total'] for res in tvmRes], index=[res['config']['n_replica'] for res in tvmRes])
        tvmSer = tvmSer.reindex(fullIndex)
        df[baselineName] = tvmSer

        kaasRes = [res for res in modelRes if res['config']['model_type'] == "kaas"]
        kaasSer = pd.Series([res[metric] for res in kaasRes], index=[res['config']['n_replica'] for res in kaasRes])
        # kaasSer = pd.Series([res[metric] for res in kaasRes], index=[res['config']['n_replica'] for res in kaasRes], dtype=np.float64)
        kaasSer = kaasSer.reindex(fullIndex)
        df[kaasName] = kaasSer

        resDfs[model] = df

    for oldName, newName in modelRenames.items():
        if oldName in resDfs:
            resDfs[newName] = resDfs[oldName]
            del resDfs[oldName]

    return resDfs


def cleanAndMergeRuns(runs):
    """Load a list or results dictionaries from multiple runs of an MlPerf
    experiment and merge them into a single dictionary with averaged submission
    and completion rates and all event lists merged"""
    for run in runs:
        if 'runTime' in run['config']:
            run['config']['t_total'] = run['config']['runTime'] * 1000
        else:
            run['config']['t_total'] = float('nan')

        if 'completion_rate' in run['metrics_warm']:
            run['completion_rate'] = run['metrics_warm']['completion_rate']['mean']
            run['submission_rate'] = run['metrics_warm']['submission_rate']['mean']
        else:
            run['completion_rate'] = run['metrics_warm']['n_completed']['mean']
            run['submission_rate'] = run['metrics_warm']['n_scheduled']['mean']

    merged = {}

    # Assume they all have the same config
    merged['config'] = copy.deepcopy(runs[0]['config'])

    # This will get re-added in the merge loop
    merged['config']['t_total'] = 0

    sRates = []
    cRates = []
    for run in runs:
        merged['config']['t_total'] += run['config']['t_total']
        sRates.append(run['submission_rate'])
        cRates.append(run['completion_rate'])

    merged['submission_rate'] = sum(sRates) / len(sRates)
    merged['completion_rate'] = sum(cRates) / len(cRates)

    merged['latencies'] = np.array(list(itertools.chain.from_iterable([run['metrics_warm']['t_response']['events'] for run in runs])))

    return merged


def loadOneMlPerf(resDirs):
    """Consolidate results from a list of mlperf output directories into a
    single pandas dataframe representing the aggregate performance of all
    results dirs.
    """
    expRuns = collections.defaultdict(list)
    for resDir in resDirs:
        for resFile in resDir.glob("*_results.json"):
            with open(resFile, 'r') as f:
                expRuns[resFile.name].append(json.load(f))

    resDicts = []
    for runs in expRuns.values():
        resDicts.append(cleanAndMergeRuns(runs))

    aggDict = {}

    # Configs are the same across all replicas
    aggDict['config'] = resDicts[0]['config']
    aggDict['config']['n_replica'] = len(resDicts)

    aggDict['completion_rate'] = 0
    aggDict['submission_rate'] = 0
    for res in resDicts:
        aggDict['completion_rate'] += res['completion_rate']
        aggDict['submission_rate'] += res['submission_rate']

    aggDict['latencies'] = np.array(list(itertools.chain.from_iterable([res['latencies'] for res in resDicts])))

    aggDict['max'] = np.max(aggDict['latencies'])
    aggDict['min'] = np.min(aggDict['latencies'])
    aggDict['mean'] = np.mean(aggDict['latencies'])
    aggDict['p10'] = np.quantile(aggDict['latencies'], 0.10)
    aggDict['p50'] = np.quantile(aggDict['latencies'], 0.50)
    aggDict['p90'] = np.quantile(aggDict['latencies'], 0.90)
    aggDict['p99'] = np.quantile(aggDict['latencies'], 0.99)
    aggDict['std'] = np.std(aggDict['latencies'])
    aggDict['cov'] = aggDict['std'] / aggDict['mean']

    aggDict['n_sample_total'] = len(aggDict['latencies'])
    aggDict['n_mean_sample_per_client'] = aggDict['n_sample_total'] / aggDict['config']['n_replica']

    return aggDict


def loadOneThroughput(resPath):
    resDicts = []
    for resFile in resPath.glob("*_results.json"):
        with open(resFile, 'r') as f:
            # resDicts += json.load(f)
            resDicts.append(json.load(f))

    aggDict = {}
    aggDict['config'] = resDicts[0]['config']
    aggDict['config']['n_replica'] = len(resDicts)

    aggDict['throughput'] = sum([d['metrics_warm']['throughput']['mean'] for d in resDicts])

    # Standard deviation between replicas
    aggDict['std'] = np.std(np.array([d['metrics_warm']['throughput']['mean'] for d in resDicts]))

    return aggDict, resDicts


def getRunDirs(resPath, expNames=None):
    """Some experiments support multiple runs that can be aggregated together
    later. These should have the form of topLevel/run0/,...,runN/ where each
    runN/ directory contains experiment results directories with the format
    [model]_[mode]_[nReplica]/. This function parses this directory structure
    and returns a dictionary mapping the experiment name to the list of
    directories containing results for that client. In other words, it flattens
    the multiple runs into lists of directories for each result.

    Returns:
        {[model]_[mode]_[nReplica]: [pathlib.Path(run0/[model]_[mode]_[nReplica]/), ... pathlib.Path(run1/...)]
    """
    expDirs = collections.defaultdict(list)
    for runDir in resPath.glob("*"):
        for resDir in runDir.glob("*"):
            if expNames is None or resDir.name in expNames:
                expDirs[resDir.name].append(resDir)

    return expDirs


def loadAllThroughput(resPath):
    fullResults = {}
    for resDir in resPath.glob("*"):
        aggRes, _ = loadOneThroughput(resDir)
        fullResults[resDir.name] = aggRes

    return aggregateModels(fullResults, 'throughput')


def loadAllMlPerf(resPath, expNames=None):
    """Load multiple runs of a suite of mlPerf experiments.

    Arguments:
        resPath: should have the form (run0/, run1/, ...) where each run has a
            list of [model]_[mode]_[nReplica] results directories, each
            containing the standard mlperf experiments results.
        expNames: a list of [model]_[mode]_[nReplica] names to output. Defaults
            to all available experiments.
    Returns:
        {"model_mode_nReplica": pd.DataFrame} # see loadOneMlPerf for details of the dataframe
    """
    # per-(model,mode,nReplica) lists of results directories
    expDirs = getRunDirs(resPath, expNames=expNames)

    # Results of a single experiment aggregated across all runs of that
    # experiment
    # {"model_mode_nReplica": pd.DataFrame}
    fullResults = {}
    for name, dirs in expDirs.items():
        aggRes = loadOneMlPerf(dirs)
        fullResults[name] = aggRes

    return fullResults


def minMaxThroughput(thrReport):
    minmax = {}
    for name, df in thrReport.items():
        minmax[name] = list(df.min(axis=1, skipna=False))

    return minmax


def getMaxThroughputs(thrReport):
    maxThr = {}
    for name, df in thrReport.items():
        maxThr[name] = {}
        maxThr[name][baselineName] = list(df.Actor)
        maxThr[name][kaasName] = list(df.KaaS)

    return maxThr


def loadOneNShot(resPath):
    with open(resPath, 'r') as f:
        allRes = json.load(f)

    for res in allRes:
        res['metrics'] = {k: v['mean'] for k, v in res['metrics'].items()}
    pprint(allRes[0]['config'])


def loadMicroNative(builtinMetrics, nvMetrics):
    metrics = {}

    kernelMetrics = nvMetrics.reset_index()
    nameCond = kernelMetrics['Name'].apply(lambda x: x[0] != '[')
    typeCond = kernelMetrics['Type'] == "GPU activities"
    kernelMetrics = kernelMetrics[nameCond & typeCond]

    metrics['t_kernel'] = kernelMetrics['Time'].sum()

    nvTimes = nvMetrics['Time']
    metrics['t_cudaMM'] = nvTimes.get('cuMemAlloc', 0.0)
    metrics['t_cudaMM'] += nvTimes.get('cudaMalloc', 0.0)
    metrics['t_cudaMM'] += nvTimes.get('cuMemsetD8', 0.0)

    metrics['t_kernel_init'] = nvTimes.get('cuModuleLoad', 0.0)
    metrics['t_kernel_init'] += nvTimes.get('cuModuleLoadData', 0.0)
    metrics['t_kernel_init'] += nvTimes.get('cudaSetDevice', 0.0)
    metrics['t_kernel_init'] += nvTimes.get('cuDeviceTotalMem', 0.0)

    metrics['t_cuda_copy'] = nvTimes.get('cuMemcpyDtoH', 0.0)
    metrics['t_cuda_copy'] += nvTimes.get('cuMemcpyHtoD', 0.0)
    metrics['t_cuda_copy'] += nvTimes.get('cudaMemcpy', 0.0)

    workerMetrics = builtinMetrics['workers']['groups']['test']
    metrics['t_data_layer'] = workerMetrics['t_loadInput']['mean']
    metrics['t_data_layer'] += workerMetrics['t_writeOutput']['mean']

    tRun = builtinMetrics['server']['t_run']['mean']
    metrics['t_other'] = tRun - sum(metrics.values())
    metrics['t_e2e'] = tRun

    return pd.Series(metrics)


def loadMicroKaas(raw):
    kaasRaw = raw['workers']['groups']['test']['kaas']

    metrics = {}
    metrics['t_kernel'] = kaasRaw['t_invoke']['mean']
    metrics['t_cudaMM'] = kaasRaw['t_cudaMM']['mean']
    metrics['t_kernel_init'] = kaasRaw['t_kernelLoad']['mean']

    metrics['t_cuda_copy'] = kaasRaw['t_dtoh']['mean']
    metrics['t_cuda_copy'] += kaasRaw['t_htod']['mean']

    metrics['t_data_layer'] = kaasRaw['t_hostDLoad']['mean']
    metrics['t_data_layer'] += kaasRaw['t_hostDWriteBack']['mean']

    tRun = raw['server']['t_run']['mean']
    metrics['t_other'] = tRun - sum(metrics.values())
    metrics['t_e2e'] = tRun

    return pd.Series(metrics)


def loadNvProf(resPath):
    with open(resPath, 'r') as f:
        dirtyLines = f.readlines()

    # NVProf sucks and produces invalid CSVs that are so bad we can't clean
    # them with pandas' builtin stuff. Gotta manually strip out the garbage.
    headerIdx = None
    for idx, line in enumerate(dirtyLines):
        if line[0:6] != '"Type"':
            continue
        else:
            headerIdx = idx

    cleanLines = []
    cleanLines.append(dirtyLines[headerIdx])
    types = dirtyLines[headerIdx + 1].split(',')
    cleanLines += dirtyLines[headerIdx + 2:]

    raw = io.StringIO('\n'.join(cleanLines))

    df = pd.read_csv(raw).set_index('Name')

    # us -> ms
    for i, t in enumerate(types):
        if t not in ['us', 'ms', '%', '', '\n']:
            print("Unrecognized type: ", repr(t))

        if t == 'us':
            df.iloc[:, i] /= 1000

    return df


def loadMicroSuiteKaas(resDir):
    resDir = resDir / 'kaas'
    coldAgg = pd.DataFrame()
    warmAgg = pd.DataFrame()
    for resPath in resDir.glob("*.json"):
        with open(resPath, 'r') as f:
            kaasNative = json.load(f)

        kaasCold = loadMicroKaas(kaasNative['metrics_cold'])
        kaasWarm = loadMicroKaas(kaasNative['metrics_warm'])

        coldAgg = pd.concat((coldAgg, kaasCold), ignore_index=True, axis=1)
        warmAgg = pd.concat((warmAgg, kaasWarm), ignore_index=True, axis=1)

    meanDf = pd.DataFrame.from_dict({"kaasWarm": warmAgg.mean(axis=1), "kaasCold": coldAgg.mean(axis=1)})
    stdDf = pd.DataFrame.from_dict({"kaasWarm": warmAgg.std(axis=1), "kaasCold": coldAgg.std(axis=1)})

    return (meanDf, stdDf)


def loadMicroSuiteNative(resDir):
    coldAgg = pd.DataFrame()
    warmAgg = pd.DataFrame()

    nvColds = []
    for resPath in (resDir / 'actNvCold').glob("*.csv"):
        # nvColds.append(loadNvProf(resPath)['Time'])
        nvColds.append(loadNvProf(resPath))

    nvWarms = []
    for resPath in (resDir / 'actNvWarm').glob("*.csv"):
        # nvWarms.append(loadNvProf(resPath)['Time'])
        nvWarms.append(loadNvProf(resPath))

    builtinColds = []
    builtinWarms = []
    for resPath in (resDir / "actPipe").glob("*.json"):
        with open(resPath, 'r') as f:
            actPipeNative = json.load(f)

        builtinColds.append(actPipeNative['metrics_cold'])
        builtinWarms.append(actPipeNative['metrics_warm'])

    for nv, builtin in zip(nvColds, builtinColds):
        coldAgg = pd.concat((coldAgg, loadMicroNative(builtin, nv)), ignore_index=True, axis=1)

    for nv, builtin in zip(nvWarms, builtinWarms):
        warmAgg = pd.concat((warmAgg, loadMicroNative(builtin, nv)), ignore_index=True, axis=1)

    meanDf = pd.DataFrame.from_dict({"actWarm": warmAgg.mean(axis=1), "actCold": coldAgg.mean(axis=1)})
    stdDf = pd.DataFrame.from_dict({"actWarm": warmAgg.std(axis=1), "actCold": coldAgg.std(axis=1)})

    return (meanDf, stdDf)


def loadMicroSuite(resDir):
    kaasMeans, kaasStds = loadMicroSuiteKaas(resDir)
    nativeMeans, nativeStds = loadMicroSuiteNative(resDir)

    means = pd.concat([kaasMeans, nativeMeans], axis=1)
    stds = pd.concat([kaasStds, nativeStds], axis=1)

    return (means, stds)


if __name__ == "__main__":
    resPath = pathlib.Path(sys.argv[1])

    full = loadAllMlPerf(resPath)
    pprint(aggregateModels(full, metric='p50'))
