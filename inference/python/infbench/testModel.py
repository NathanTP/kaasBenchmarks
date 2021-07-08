from . import model
from . import dataset

import time
import numpy as np

# nelem for one side of the matrices (all are square)
matSize = 2

preTime = 0
runTime = 0
postTime = 0


class testModel(model.Model):
    # Standard Parameters
    nOutPre = 1
    preMap = model.inputMap(const=(0,), inp=(0,))

    nOutRun = 1
    runMap = model.inputMap(const=(0,), pre=(0,))

    nOutPost = 1
    postMap = model.inputMap(const=(0,), run=(0,))

    nConst = 1
    noPost = False

    def __init__(self, modelDir):
        pass

    @staticmethod
    def getConstants(modelDir):
        const = np.arange(0, matSize**2, 1, dtype=np.float32)
        const.shape = (matSize, matSize)
        return (const,)

    @staticmethod
    def pre(data):
        result = data[1] + 1
        return (result,)

    def run(self, data):
        const = data[0]
        inp = data[1]

        time.sleep(runTime)
        result = np.matmul(const, inp)
        return (result,)

    @staticmethod
    def post(data):
        result = data[1] - 1
        return (result,)

    @staticmethod
    def getMlPerfCfg(testing=False):
        settings = model.getDefaultMlPerfCfg()

        totalDelay = sum(preTime, runTime, postTime)
        if totalDelay == 0:
            settings.server_target_qps = 100
        else:
            settings.server_target_qps = (1 / (preTime + runTime + postTime)) / 2

        return settings


class testLoader(dataset.loader):
    # This is arbitrary
    ndata = 1000
    checkAvailable = True

    def __init__(self, dataDir):
        self.data = {}

    def preLoad(self, idxs):
        for i in idxs:
            self.data[i] = np.full((matSize, matSize), (i+1)*10, dtype=np.float32)

    def unLoad(self, idxs):
        for i in idxs:
            del self.data[i]

    def get(self, idx):
        return (self.data[idx],)

    def check(self, result, idx):
        result = result[0]
        expect = self.data[idx]
        expect += 1
        expect = np.matmul(testModel.getConstants(None), expect)
        expect -= 1

        return np.allclose(result, expect, rtol=0.05, atol=0)
