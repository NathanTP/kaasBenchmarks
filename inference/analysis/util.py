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


def loadKaasNShot(res):
    clean = {}


def loadOneNShot(resPath):
    with open(resPath, 'r') as f:
        allRes = json.load(f)

    for res in allRes:
        res['metrics'] = {k: v['mean'] for k, v in res['metrics'].items()}
    pprint(allRes[0]['config'])


if __name__ == "__main__":
    resPath = pathlib.Path(sys.argv[1])

    loadOneNShot(resPath)
    # model = 'resnet50'

    # print(model)
    # print(loadAllMlPerf(resPath, metric="n_sample_total")[model])

    # print(getMaxThroughputs(loadAllThroughput(resPath)))
    # print(loadAllThroughput(resPath)[model])
    # print(loadAllMlPerf(resPath, metric='completion_rate')['resnet50'])

    # dirs = getRunDirs(resPath, expNames=['resnet50_tvm_5'])
    # res, _ = loadOneMlPerf(dirs['resnet50_tvm_5'])
    # print(res['submission_rate'])
