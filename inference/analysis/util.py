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
    merged['latencies'] *= 1000

    return merged


def loadOneMlPerf(resDirs):
    """Consolidate results from a single mlperf output directory into a pandas dataframe"""
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

    return aggDict, resDicts


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
    expDirs = {}
    for runDir in resPath.glob("*"):
        for resDir in runDir.glob("*"):
            if expNames is None or resDir.name in expNames:
                if resDir.name not in expDirs:
                    expDirs[resDir.name] = []

                expDirs[resDir.name].append(resDir)

    return expDirs


def loadAllThroughput(resPath):
    fullResults = {}
    for resDir in resPath.glob("*"):
        aggRes, _ = loadOneThroughput(resDir)
        fullResults[resDir.name] = aggRes

    return aggregateModels(fullResults, 'throughput')


def loadAllMlPerf(resPath, metric='p90', expNames=None):
    expDirs = getRunDirs(resPath, expNames=expNames)

    fullResults = {}
    for name, dirs in expDirs.items():
        aggRes, _ = loadOneMlPerf(dirs)
        fullResults[name] = aggRes

    return fullResults, aggregateModels(fullResults, metric)


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
    builtinMetrics = {metric: val['mean'] for metric, val in builtinMetrics.items()}

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

    metrics['t_data_layer'] = builtinMetrics['t_loadInput']
    metrics['t_data_layer'] += builtinMetrics['t_writeOutput']

    metrics['t_other'] = builtinMetrics['t_run'] - sum(metrics.values())
    metrics['t_e2e'] = builtinMetrics['t_run']

    return pd.Series(metrics)


def loadMicroKaas(builtinMetrics):
    builtinMetrics = {metric: val['mean'] for metric, val in builtinMetrics.items()}

    metrics = {}
    metrics['t_kernel'] = builtinMetrics['kaas:t_invoke']
    metrics['t_cudaMM'] = builtinMetrics['kaas:t_cudaMM']
    metrics['t_kernel_init'] = builtinMetrics['kaas:t_kernelLoad']

    metrics['t_cuda_copy'] = builtinMetrics['kaas:t_dtoh']
    metrics['t_cuda_copy'] += builtinMetrics['kaas:t_htod']

    metrics['t_data_layer'] = builtinMetrics['kaas:t_hostDLoad']
    metrics['t_data_layer'] += builtinMetrics['kaas:t_hostDWriteBack']
    metrics['t_data_layer'] += builtinMetrics['kaas:t_load_request']

    metrics['t_other'] = builtinMetrics['t_run'] - sum(metrics.values())
    metrics['t_e2e'] = builtinMetrics['t_run']

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

        coldAgg = coldAgg.append(kaasCold, ignore_index=True)
        warmAgg = warmAgg.append(kaasWarm, ignore_index=True)

    meanDf = pd.DataFrame.from_dict({"kaasWarm": warmAgg.mean(), "kaasCold": coldAgg.mean()})
    stdDf = pd.DataFrame.from_dict({"kaasWarm": warmAgg.std(), "kaasCold": coldAgg.std()})

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
        coldAgg = coldAgg.append(loadMicroNative(builtin, nv), ignore_index=True)

    for nv, builtin in zip(nvWarms, builtinWarms):
        warmAgg = warmAgg.append(loadMicroNative(builtin, nv), ignore_index=True)

    meanDf = pd.DataFrame.from_dict({"actWarm": warmAgg.mean(), "actCold": coldAgg.mean()})
    stdDf = pd.DataFrame.from_dict({"actWarm": warmAgg.std(), "actCold": coldAgg.std()})

    return (meanDf, stdDf)


def loadMicroSuite(resDir):
    kaasMeans, kaasStds = loadMicroSuiteKaas(resDir)
    nativeMeans, nativeStds = loadMicroSuiteNative(resDir)

    means = pd.concat([kaasMeans, nativeMeans], axis=1)
    stds = pd.concat([kaasStds, nativeStds], axis=1)

    return (means, stds)


if __name__ == "__main__":
    resPath = pathlib.Path(sys.argv[1])

    # print(loadNvProf(resPath / 'actNvWarm' / '0_results.csv'))

    # means, stds = loadMicroSuite(resPath)
    # print(means)

    # print(loadMicro(resPath))
    # loadOneNShot(resPath)
    # model = 'resnet50'

    # print(getMaxThroughputs(loadAllThroughput(resPath)))
    # print(loadAllThroughput(resPath)[model])
    # print(loadAllMlPerf(resPath, metric='n_sample_total')['cGEMM'])

    # dirs = getRunDirs(resPath, expNames=['resnet50_tvm_5'])
    # res, _ = loadOneMlPerf(dirs['resnet50_tvm_5'])
    # res, _ = loadOneMlPerf([resPath])

    # print(loadAllThroughput(resPath))
    full, agg = loadAllMlPerf(resPath, metric="p50")
    print(agg['resnet50'])
    # # updateFormat(resPath, pathlib.Path(sys.argv[2]), suiteType='throughput')
    # updateFormat(resPath, pathlib.Path(sys.argv[2]), suiteType='mlperf')
