import io
import json
import pathlib
import itertools
import copy
import sys
import pandas as pd
import numpy as np
import collections
from pprint import pprint


def aggregateModels(fullResults, metric):
    models = ['resnet50', 'bert', 'jacobi']
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
        df['Actor'] = tvmSer

        kaasRes = [res for res in modelRes if res['config']['model_type'] == "kaas"]
        kaasSer = pd.Series([res[metric] for res in kaasRes], index=[res['config']['n_replica'] for res in kaasRes], dtype=np.float64)
        kaasSer = kaasSer.reindex(fullIndex)
        df['KaaS'] = kaasSer

        resDfs[model] = df

    return resDfs


def cleanAndMergeRuns(runs):
    for run in runs:
        if 'runTime' in run['config']:
            run['config']['t_total'] = run['config']['runTime'] * 1000
        else:
            run['config']['t_total'] = float('nan')

        if 'completion_rate' in run['metrics']:
            run['completion_rate'] = run['metrics']['completion_rate']
            run['submission_rate'] = run['metrics']['submission_rate']
        else:
            run['completion_rate'] = run['metrics']['n_completed']
            run['submission_rate'] = run['metrics']['n_scheduled']

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

    merged['latencies'] = np.array(list(itertools.chain.from_iterable([run['metrics']['latencies'] for run in runs])))
    merged['latencies'] *= 1000

    return merged


def loadOneMlPerf(resDirs):
    """Consolidate results from a single mlperf output directory into a pandas dataframe"""
    expRuns = collections.defaultdict(list)
    for resDir in resDirs:
        for resFile in resDir.glob("*_results.json"):
            with open(resFile, 'r') as f:
                expRuns[resFile.name].append(json.load(f)[0])

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
            resDicts += json.load(f)

    aggDict = {}
    aggDict['config'] = resDicts[0]['config']
    aggDict['config']['n_replica'] = len(resDicts)

    aggDict['throughput'] = sum(d['metrics']['throughput'] for d in resDicts)
    # Standard deviation between replicas
    aggDict['std'] = np.std(np.array([d['metrics']['throughput'] for d in resDicts]))

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

    return aggregateModels(fullResults, metric)


def minMaxThroughput(thrReport):
    minmax = {}
    for name, df in thrReport.items():
        minmax[name] = list(df.min(axis=1, skipna=False))

    return minmax


def getMaxThroughputs(thrReport):
    maxThr = {}
    for name, df in thrReport.items():
        maxThr[name] = {}
        maxThr[name]['Actor'] = list(df.Actor)
        maxThr[name]['KaaS'] = list(df.KaaS)

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

    metrics['t_kernel'] = nvMetrics.get('sgemm', 0.0)
    metrics['t_cudaMM'] = nvMetrics.get('cuMemAlloc', 0.0)
    metrics['t_cudaMM'] += nvMetrics.get('cuMemsetD8', 0.0)
    metrics['t_kernel_init'] = nvMetrics.get('cuModuleLoad', 0.0)
    metrics['t_cuda_copy'] = nvMetrics.get('cuMemcpyDtoH', 0.0)
    metrics['t_cuda_copy'] += nvMetrics.get('cuMemcpyHtoD', 0.0)

    metrics['t_data_layer'] = builtinMetrics['t_loadInput']
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

    metrics['t_other'] = builtinMetrics['t_run'] - sum(metrics.values())
    metrics['t_e2e'] = builtinMetrics['t_run']

    return pd.Series(metrics)


def loadNvProf(resPath):
    with open(resPath, 'r') as f:
        dirtyLines = f.readlines()

    # NVProf sucks and produces invalid CSVs that are so bad we can't clean
    # them with pandas' builtin stuff. Gotta manually strip out the garbage.
    cleanLines = []
    for line in dirtyLines:
        if line[0] == ',':
            types = line.split(',')
        elif line[0] != '=':
            cleanLines.append(line)

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
        nvColds.append(loadNvProf(resPath)['Time'])

    nvWarms = []
    for resPath in (resDir / 'actNvWarm').glob("*.csv"):
        nvWarms.append(loadNvProf(resPath)['Time'])

    builtinColds = []
    builtinWarms = []
    for resPath in (resDir / "actInline").glob("*.json"):
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
    means, stds = loadMicroSuite(resPath)
    print(stds)

    # print(loadMicro(resPath))
    # loadOneNShot(resPath)
    # model = 'resnet50'

    # print(model)
    # print(loadAllMlPerf(resPath, metric="n_sample_total")[model])

    # print(getMaxThroughputs(loadAllThroughput(resPath)))
    # print(loadAllThroughput(resPath)[model])
    # print(loadAllMlPerf(resPath, metric='completion_rate')['resnet50'])

    # dirs = getRunDirs(resPath, expNames=['resnet50_tvm_5'])
    # res, _ = loadOneMlPerf(dirs['resnet50_tvm_5'])
    # print(res['submission_rate'])
