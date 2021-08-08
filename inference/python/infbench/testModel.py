from . import model
from . import dataset

import time
import numpy as np
import pickle

import pycuda.driver as cuda
import pycuda.tools

# These parameters should match kaasSources/sgemm to be consistent, though if you
# only want to run testModelNP they can be anything you want.
matSize = 128  # side length of test matrices (all square)
depth = 3  # number of chained multiplies to use

preTime = 0
runTime = 0.01
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


class testModelNP(testModel, model.Model):
    """A numpy-based model"""

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

    def run(self, data):
        constants = data[:self.nConst]
        inputs = data[self.nConst:]

        time.sleep(runTime)

        expect = np.matmul(inputs[0], constants[0])
        for i in range(1, depth):
            expect = np.matmul(expect, constants[i])

        return (expect,)

    @staticmethod
    def getMlPerfCfg(gpuType, benchConfig):
        if gpuType == "Tesla K20c":
            maxQps = 100
            medianLatency = 0.025
        elif gpuType == "Tesla V100-SXM2-16GB":
            maxQps = 50
            medianLatency = 0.025
        else:
            raise ValueError("Unrecoginzied GPU Type" + gpuType)

        settings = model.getDefaultMlPerfCfg(maxQps, medianLatency, benchConfig)

        return settings


class testModelNative(testModel, model.Model):
    """Calls the GPU kernel natively instead of using KaaS"""

    def __init__(self, modelArg):
        super().__init__(modelArg)
        self.modelPath = modelArg
        self.dConsts = None
        self.dIOs = None

        tile_tb_height = 8
        tileN = 16
        tileM = (tileN * tile_tb_height)

        # Size of one element in bytes, e.g. float32=4
        self.gridDim = (matSize // tileM, matSize // tileN, 1)
        self.blockDim = (tileN, tile_tb_height, 1)
        self.sharedSize = tile_tb_height * tileN * 4

        cuda.init()
        self.cudaCtx = pycuda.tools.make_default_context()

        mod = cuda.module_from_file(str(self.modelPath.parent / "sgemm.cubin"))
        self.kern = mod.get_function("sgemm")
        self.kern.prepare(["P", "P", "P"])

    def __del__(self):
        if self.dConsts is not None:
            for dConst in self.dConsts:
                dConst.free()

        if self.dIOs is not None:
            for dBuf in self.dIOs:
                dBuf.free()

        self.cudaCtx.detach()

    @staticmethod
    def getConstants(modelDir):
        with open(modelDir / "sgemm_params.pkl", 'rb') as f:
            constants = pickle.load(f)
        return constants

    def run(self, data):
        constants = data[:self.nConst]
        hInp = data[self.nConst]

        self.cudaCtx.push()

        if self.dConsts is None:
            self.dConsts = []
            for hConst in constants:
                dConst = cuda.mem_alloc(hConst.nbytes)
                cuda.memcpy_htod(dConst, hConst)
                self.dConsts.append(dConst)

        if self.dIOs is None:
            self.dIOs = []
            self.dIOs.append(cuda.mem_alloc(hInp.nbytes))

            for i in range(depth):
                self.dIOs.append(cuda.mem_alloc(hInp.nbytes))

        cuda.memcpy_htod(self.dIOs[0], hInp)

        for i in range(depth):
            self.kern.prepared_call(self.gridDim, self.blockDim,
                                    self.dIOs[i], self.dConsts[i], self.dIOs[i+1],
                                    shared_size=self.sharedSize)

        hRes = np.empty_like(hInp)
        cuda.memcpy_dtoh(hRes, self.dIOs[-1])

        return (hRes,)

    @staticmethod
    def getMlPerfCfg(gpuType, benchConfig):
        if gpuType == "Tesla K20c":
            maxQps = 148
            medianLatency = 0.014
        elif gpuType == "Tesla V100-SXM2-16GB":
            maxQps = 125
            medianLatency = 0.016
        else:
            raise ValueError("Unrecoginzied GPU Type" + gpuType)

        settings = model.getDefaultMlPerfCfg(maxQps, medianLatency, benchConfig)

        return settings


class testModelKaas(testModel, model.kaasModel):
    def __init__(self, modelArg):
        super().__init__(modelArg)

    @staticmethod
    def getMlPerfCfg(gpuType, benchConfig):
        if gpuType == "Tesla K20c":
            maxQps = 126
            medianLatency = 0.017
        elif gpuType == "Tesla V100-SXM2-16GB":
            maxQps = 126
            medianLatency = 0.020
        else:
            raise ValueError("Unrecoginzied GPU Type" + gpuType)

        settings = model.getDefaultMlPerfCfg(maxQps, medianLatency, benchConfig)

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
        constants = testModelNP.getConstants(None)

        # pre
        expect += 1

        # run
        expect = np.matmul(expect, constants[0])
        for i in range(1, depth):
            expect = np.matmul(expect, constants[i])

        # post
        expect -= 1

        return np.allclose(result, expect, rtol=0.05, atol=0)
