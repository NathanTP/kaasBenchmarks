from . import model
from . import dataset
from . import util

import libff.kaas as kaas


class dummyModel():
    nConst = 0
    nOutPre = 1
    preMap = model.inputMap(inp=(0,))

    nOutRun = 1
    runMap = model.inputMap(pre=(0,))

    nOutPost = 1
    postMap = model.inputMap(run=(0,))

    noPost = False

    @staticmethod
    def pre(data):
        return data

    @staticmethod
    def post(data):
        return data

    @staticmethod
    def getPerfEstimates(gpuType):
        return (1, 0.01)

    @classmethod
    def getMlPerfCfg(cls, gpuType, benchConfig):
        maxQps, medianLatency = cls.getPerfEstimates(gpuType, benchConfig)
        settings = model.getDefaultMlPerfCfg(maxQps, medianLatency, benchConfig)

        return settings


class dummyModelTvm(dummyModel):
    @staticmethod
    def run(data, stats=None):
        return data


class dummyModelKaas(dummyModel, model.kaasModel):
    def __init__(self, modelArg, *args, **kwargs):
        super().__init__(modelArg, *args, **kwargs)


class dummyLoader(dataset.loader):
    ndata = 1
    checkAvailable = True

    def __init__(self, dataDir):
        pass

    def preLoad(self, idxs):
        pass

    def unLoad(self, idxs):
        pass

    def get(self, idx):
        return bytes([1]*1024)

    def check(self, result, idx):
        return (result == [1]*1024)
