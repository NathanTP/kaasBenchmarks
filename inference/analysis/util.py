import json
import pathlib
import itertools
import sys
import pandas as pd
import numpy as np


def loadOneMlPerf(resPath):
    """Consolidate results from a single mlperf output directory into a pandas dataframe"""
    resDicts = []
    for resFile in resPath.glob("*_results.json"):
        with open(resFile, 'r') as f:
            resDicts += json.load(f)

    print(resPath)
    print(len(resDicts))
    aggDict = {}
    # Configs are the same across all replicas
    aggDict['config'] = resDicts[0]['config']
    aggDict['config']['nReplica'] = len(resDicts)

    aggDict['n_completed'] = sum([res['metrics']['n_completed'] for res in resDicts])

    aggDict['latencies'] = np.array(list(itertools.chain.from_iterable([res['metrics']['latencies'] for res in resDicts])))
    aggDict['p50'] = np.quantile(aggDict['latencies'], 0.50) * 1000
    aggDict['p90'] = np.quantile(aggDict['latencies'], 0.90) * 1000
    aggDict['p99'] = np.quantile(aggDict['latencies'], 0.99) * 1000

    return aggDict, resDicts

def loadOneThroughput(resPath):
    resDicts = []
    for resFile in resPath.glob("*_results.json"):
        with open(resFile, 'r') as f:
            resDicts += json.load(f)

    aggDict = {}
    aggDict['config'] = resDicts[0]['config']
    aggDict['throughput'] = sum(d['metrics']['throughput'] for d in resDicts)
    aggDict['config']['nReplica'] = len(resDicts)

    return aggDict, resDicts

def loadAllThroughput(resPath):
    fullResults = {}
    for resDir in resPath.glob("*"):
        aggRes, _ = loadOneThroughput(resDir)
        fullResults[resDir.name] = aggRes

    models = ['resnet50', 'bert', 'jacobi']
    resDfs = {}
    for model in models:
        modelRes = [res for res in fullResults.values() if model in res['config']['model']]
        modelRes = sorted(modelRes, key=lambda x: x['config']['nReplica'])
        tvmThroughputs = [res['throughput'] for res in modelRes if res['config']['model_type'] in ["tvm", "direct"]]
        kaasThroughputs = [res['throughput'] for res in modelRes if res['config']['model_type'] == "kaas"]

        while len(tvmThroughputs) < len(kaasThroughputs):
            tvmThroughputs.append(float('nan'))

        resDfs[model] = pd.DataFrame({'Actor': tvmThroughputs, 'KaaS': kaasThroughputs},
                                     index=range(1,len(tvmThroughputs) + 1))

    return resDfs


def loadAllMlPerf(resPath):
    fullResults = {}
    for resDir in resPath.glob("*"):
        aggRes, _ = loadOneMlPerf(resDir)
        fullResults[resDir.name] = aggRes

    models = ['resnet50', 'bert']
    resDfs = {}
    for model in models:
        modelRes = [res for res in fullResults.values() if model in res['config']['model']]
        modelRes = sorted(modelRes, key=lambda x: x['config']['nReplica'])
        tvmThroughputs = [res['n_completed'] for res in modelRes if res['config']['model_type'] in ["tvm", "direct"]]
        kaasThroughputs = [res['n_completed'] for res in modelRes if res['config']['model_type'] == "kaas"]
        # tvmThroughputs = [res['config']['scale'] for res in modelRes if res['config']['model_type'] == "tvm"]
        # kaasThroughputs = [res['config']['scale'] for res in modelRes if res['config']['model_type'] == "kaas"]

        while len(tvmThroughputs) < len(kaasThroughputs):
            tvmThroughputs.append(float('nan'))

        resDfs[model] = pd.DataFrame({'Actor': tvmThroughputs, 'KaaS': kaasThroughputs},
                                     index=range(1, len(tvmThroughputs) + 1))
    return resDfs


def loadAllLatDist(resPath):
    fullResults = {}
    for resDir in resPath.glob("*"):
        aggRes, _ = loadOneMlPerf(resDir)
        fullResults[resDir.name] = aggRes

    models = ['resnet50', 'bert']
    resDfs = {}
    for model in models:
        modelRes = [res for res in fullResults.values() if model in res['config']['model']]
        modelRes = sorted(modelRes, key=lambda x: x['config']['nReplica'])
        tvmRes = [res['p90'] for res in modelRes if res['config']['model_type'] == "tvm"]
        kaasRes = [res['p90'] for res in modelRes if res['config']['model_type'] == "kaas"]

        maxLen = max(len(tvmRes), len(kaasRes))
        for res in [tvmRes, kaasRes]:
            while len(res) < maxLen:
                res.append(float('nan'))

        resDfs[model] = pd.DataFrame({'Actor': tvmRes, 'KaaS': kaasRes},
                                     index=range(1, len(tvmRes) + 1))
    return resDfs


def minMaxThroughput(thrReport):
    minmax = {}
    for name, df in thrReport.items():
        minmax[name] = list(df.min(axis=1, skipna=False))

    return minmax


if __name__ == "__main__":
    resPath = pathlib.Path(sys.argv[1])

    model = 'jacobi'
    print(model)
    # print(loadAllLatDist(resPath)[model])
    # print(minMaxThroughput(loadAllThroughput(resPath)))
    print(loadAllThroughput(resPath)[model])
    # print(loadAllMlPerf(resPath)['resnet50'])
