import json
import pathlib
import itertools
import sys
import pandas as pd
import numpy as np


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
        tvmSer = tvmSer.reindex(fullIndex)
        df['Actor'] = tvmSer

        kaasRes = [res for res in modelRes if res['config']['model_type'] == "kaas"]
        kaasSer = pd.Series([res[metric] for res in kaasRes], index=[res['config']['n_replica'] for res in kaasRes], dtype=np.float64)
        kaasSer = kaasSer.reindex(fullIndex)
        df['KaaS'] = kaasSer

        resDfs[model] = df

    return resDfs


def loadOneMlPerf(resPath):
    """Consolidate results from a single mlperf output directory into a pandas dataframe"""
    resDicts = []
    for resFile in resPath.glob("*_results.json"):
        with open(resFile, 'r') as f:
            resDicts += json.load(f)

    aggDict = {}
    # Configs are the same across all replicas
    aggDict['config'] = resDicts[0]['config']
    aggDict['config']['n_replica'] = len(resDicts)

    aggDict['completion_rate'] = sum([res['metrics']['completion_rate'] for res in resDicts])
    # aggDict['submission_rate'] = sum([res['metrics']['submission_rate'] for res in resDicts])

    aggDict['latencies'] = np.array(list(itertools.chain.from_iterable([res['metrics']['latencies'] for res in resDicts])))
    aggDict['p50'] = np.quantile(aggDict['latencies'], 0.50) * 1000
    aggDict['p90'] = np.quantile(aggDict['latencies'], 0.90) * 1000
    aggDict['p99'] = np.quantile(aggDict['latencies'], 0.99) * 1000
    aggDict['n_sample_total'] = len(aggDict['latencies'])
    aggDict['n_mean_sample_pre_client'] = aggDict['n_sample_total'] / aggDict['config']['n_replica']

    return aggDict, resDicts


def loadOneThroughput(resPath):
    resDicts = []
    for resFile in resPath.glob("*_results.json"):
        with open(resFile, 'r') as f:
            resDicts += json.load(f)

    aggDict = {}
    aggDict['config'] = resDicts[0]['config']
    aggDict['throughput'] = sum(d['metrics']['throughput'] for d in resDicts)
    aggDict['config']['n_replica'] = len(resDicts)

    return aggDict, resDicts


def loadAllThroughput(resPath):
    fullResults = {}
    for resDir in resPath.glob("*"):
        aggRes, _ = loadOneThroughput(resDir)
        fullResults[resDir.name] = aggRes

    return aggregateModels(fullResults, 'throughput')


def loadAllMlPerf(resPath):
    fullResults = {}
    for resDir in resPath.glob("*"):
        aggRes, _ = loadOneMlPerf(resDir)
        fullResults[resDir.name] = aggRes

    return aggregateModels(fullResults, 'completion_rate')


def loadAllLatDist(resPath):
    fullResults = {}
    for resDir in resPath.glob("*"):
        aggRes, _ = loadOneMlPerf(resDir)
        fullResults[resDir.name] = aggRes

    return aggregateModels(fullResults, 'p90')


def minMaxThroughput(thrReport):
    minmax = {}
    for name, df in thrReport.items():
        minmax[name] = list(df.min(axis=1, skipna=False))

    return minmax


if __name__ == "__main__":
    resPath = pathlib.Path(sys.argv[1])

    model = 'bert'
    print(model)
    print(loadAllLatDist(resPath)[model])
    # print(minMaxThroughput(loadAllThroughput(resPath)))
    # print(loadAllThroughput(resPath)[model])
    # print(loadAllMlPerf(resPath)['resnet50'])
