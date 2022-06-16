from . import model
from . import dataset

from kaas import profiling

import math
import time
import numpy as np
import pickle

import pycuda.driver as cuda
import pycuda.tools


# These parameters should match kaasSources/sgemm to be consistent, though if you
# only want to run testModelNP they can be anything you want.
matSize = 1024  # side length of test matrices (all square)
depth = 3  # number of chained multiplies to use

preTime = 20
postTime = 20


class testModel():
    # Standard Parameters
    nConst = 1

    nOutPre = 1
    preMap = model.inputMap(inp=(0,))

    nOutRun = 1
    runMap = model.inputMap(const=(0,), pre=(0,))

    nOutPost = 1
    postMap = model.inputMap(run=(0,))

    noPost = False

    # This acts like a mixin, see
    # https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def pre(data):
        result = data[0]
        if isinstance(result, bytes):
            result = np.frombuffer(result, dtype=np.float32)
            result.shape = (matSize, matSize)

        result = result + 1
        time.sleep(preTime / 1000)
        return (result,)

    @staticmethod
    def post(data):
        inputArr = data[0]
        inputArr.dtype = np.float32
        inputArr.shape = (matSize, matSize)

        result = inputArr - 1

        time.sleep(postTime / 1000)

        return (result,)

    @staticmethod
    def getPerfEstimates(gpuType):
        if gpuType == "Tesla K20c":
            maxQps = 18
            medianLatency = 0.100
        elif gpuType == "Tesla V100-SXM2-16GB":
            maxQps = 125
            medianLatency = 0.016
        else:
            raise ValueError("Unrecoginzied GPU Type" + gpuType)

        return maxQps, medianLatency

    @classmethod
    def getMlPerfCfg(cls, gpuType, benchConfig):
        maxQps, medianLatency = cls.getPerfEstimates(gpuType)
        settings = model.getDefaultMlPerfCfg(maxQps, medianLatency, benchConfig)

        return settings


class testModelNative(testModel, model.Model):
    """Calls the GPU kernel natively instead of using KaaS"""

    def __init__(self, modelArg):
        super().__init__(modelArg)
        self.modelPath = modelArg
        self.dConsts = None
        self.dIOs = None

        cuda.init()
        self.cudaCtx = pycuda.tools.make_default_context()
        profiling.cudaProfilerResetCtx()

        gpuHandle = cuda.Device(0)
        maxThreads = gpuHandle.get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK)

        tileSize = int(math.sqrt(maxThreads))
        self.gridDim = (matSize // tileSize, matSize // tileSize, 1)
        self.blockDim = (tileSize, tileSize, 1)
        self.sharedSize = 0

        mod = cuda.module_from_file(str(self.modelPath.parent / "sgemm.cubin"))
        self.kern = mod.get_function("sgemm")
        self.kern.prepare(["i", "P", "P", "P"])

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

    def run(self, data, stats=None):
        packedConstants = data[:self.nConst]
        hInp = data[self.nConst]
        inpSize = hInp.nbytes

        constants = []
        nElem = matSize ** 2
        for i in range(depth):
            start = i * nElem
            end = (i+1) * nElem
            constArr = packedConstants[0][start:end]
            constArr.shape = (matSize, matSize)
            constants.append(constArr)

        if self.dConsts is None:
            self.dConsts = []
            for hConst in constants:
                dConst = cuda.mem_alloc(inpSize)
                cuda.memcpy_htod(dConst, hConst)
                self.dConsts.append(dConst)

        if self.dIOs is None:
            self.dIOs = []
            self.dIOs.append(cuda.mem_alloc(inpSize))

            for i in range(depth):
                self.dIOs.append(cuda.mem_alloc(inpSize))

        for i in range(1, depth + 1):
            cuda.memset_d8(self.dIOs[i], 0, inpSize)

        cuda.memcpy_htod(self.dIOs[0], hInp)

        for i in range(depth):
            self.kern.prepared_call(self.gridDim, self.blockDim,
                                    matSize,
                                    self.dIOs[i], self.dConsts[i], self.dIOs[i+1],
                                    shared_size=self.sharedSize)

        hRes = np.empty(inpSize, dtype=np.int8)
        cuda.memcpy_dtoh(hRes, self.dIOs[-1])

        return (hRes,)


class testModelKaas(testModel, model.kaasModel):
    def __init__(self, modelArg, *args, **kwargs):
        super().__init__(modelArg, *args, **kwargs)


class testLoader(dataset.loader):
    # This is arbitrary
    ndata = 1000
    checkAvailable = True

    def __init__(self, dataDir):
        with open(dataDir / "sgemm_params.pkl", 'rb') as f:
            self.constants = pickle.load(f)

        self.data = {}

    def preLoad(self, idxs):
        rng = np.random.default_rng(1)
        for i in idxs:
            self.data[i] = rng.standard_normal((matSize, matSize), dtype=np.float32)

    def unLoad(self, idxs):
        for i in idxs:
            del self.data[i]

    def get(self, idx):
        return [self.data[idx]]

    def check(self, result, idx):
        result = result[0]
        if isinstance(result, bytes):
            result = np.frombuffer(result, dtype=np.float32)
            result.shape = (matSize, matSize)

        expect = np.frombuffer(self.data[idx], dtype=np.float32)
        expect.shape = (matSize, matSize)

        # pre
        expect += 1

        # run
        constants = []
        nElem = matSize ** 2
        for i in range(depth):
            start = i * nElem
            end = (i+1) * nElem
            constArr = self.constants[0][start:end]
            constArr.shape = (matSize, matSize)
            constants.append(constArr)

        for i in range(1, depth):
            expect = np.matmul(expect, constants[i])

        # post
        expect -= 1

        return np.allclose(result, expect, rtol=0.05, atol=0)
