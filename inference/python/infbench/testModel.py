from . import model
from . import dataset

import time
import numpy as np

# These parameters should match kaasSources/sgemm to be consistent, though if you
# only want to run testModelNP they can be anything you want.
matSize = 128  # side length of test matrices (all square)
depth = 3  # number of chained multiplies to use

preTime = 0
runTime = 0
postTime = 0


class testModel():
    # Standard Parameters
    nConst = depth

    nOutPre = 1
    preMap = model.inputMap(const=(0,), inp=(0,))

    nOutRun = 1
    runMap = model.inputMap(const=range(depth), pre=(0,))

    nOutPost = 1
    postMap = model.inputMap(const=(0,), run=(0,))

    noPost = False

    # This acts like a mixin, see
    # https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def getConstants(modelDir):
        """For easy debugging, constants for the test are just a matrix filled
        with the 1-indexed index"""
        consts = []
        for i in range(depth):
            const = np.zeros((matSize, matSize), dtype=np.float32)
            np.fill_diagonal(const, i+1)
            consts.append(const)
        return consts

    @staticmethod
    def pre(data):
        result = data[1] + 1
        return (result,)

    @staticmethod
    def post(data):
        inputArr = data[1]
        inputArr.dtype = np.float32
        inputArr.shape = (matSize, matSize)
        result = inputArr - 1

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


class testModelNP(testModel, model.Model):
    """A numpy-based model"""

    def run(self, data):
        constants = data[:self.nConst]
        inputs = data[self.nConst:]

        time.sleep(runTime)

        expect = np.matmul(inputs[0], constants[0])
        for i in range(1, depth):
            expect = np.matmul(expect, constants[i])

        return (expect,)


class testModelKaas(testModel, model.kaasModel):
    def __init__(self, modelArg):
        super().__init__(modelArg)


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
        constants = testModel.getConstants(None)

        # pre
        expect += 1

        # run
        expect = np.matmul(expect, constants[0])
        for i in range(1, depth):
            expect = np.matmul(expect, constants[i])

        # post
        expect -= 1

        return np.allclose(result, expect, rtol=0.05, atol=0)
