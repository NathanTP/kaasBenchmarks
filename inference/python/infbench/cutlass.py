from . import model
from . import dataset
import numpy as np
import yaml


class sgemmBase(model.Model):
    noPost = True
    preMap = model.inputMap(inp=(0, 1))
    runMap = model.inputMap(pre=(0, 1))
    postMap = model.inputMap(run=(0,))
    nOutRun = 1
    nOutPre = 2
    nOutPost = 1
    nConst = 0

    @staticmethod
    def pre(imgBuf):
        return imgBuf

    @staticmethod
    def post(label):
        raise AttributeError("cutlass sgemm has no post-processing")

    @staticmethod
    def getConstants(modelDir):
        return []


class sgemm(sgemmBase):
    def run(self, dat):
        """Run the model against input 'dat'. Dat is expected to be a bytes
       object that can be converted to numpy/tvm and passed to the model as
       input."""
        pass

    @staticmethod
    def getMlPerfCfg(gpuType, benchConfig):
        if gpuType == "Tesla K20c":
            maxQps = 0
            medianLatency = 0.07
        elif gpuType == "Tesla V100-SXM2-16GB":
            maxQps = 0
            medianLatency = 0.05
        else:
            raise ValueError("Unrecoginzied GPU Type" + gpuType)

        settings = model.getDefaultMlPerfCfg(maxQps, medianLatency, benchConfig)

        return settings


class sgemmKaas(sgemmBase, model.kaasModel):
    def __init__(self, modelArg):
        """Can be initialized either by an existing kaasModel or by a path to a
        KaaS model. If a path is passed, it should be a directory containing:
        name.cubin, name_meta.yaml, and name_model.yaml (where name is the
        name of the directory)."""
        # In some cases, it's easier to pass a pre-initialized model as an
        # argument, typically to keep abstractions clean on the client side.
        if isinstance(modelArg, model.kaasModel):
            self.cubin = modelArg.cubin
            self.reqTemplate = modelArg.reqTemplate
            self.meta = modelArg.meta
        else:
            modelDir = modelArg.parent

            baseName = modelDir.stem
            self.cubin = modelDir / (baseName + ".cubin")
            with open(modelDir / (baseName + "_model" + ".yaml"), 'r') as f:
                self.reqTemplate = yaml.safe_load(f)

            with open(modelDir / (baseName + "_meta" + ".yaml"), 'r') as f:
                self.meta = yaml.safe_load(f)

    @staticmethod
    def getMlPerfCfg(gpuType, benchConfig):
        if gpuType == "Tesla K20c":
            maxQps = 0
            medianLatency = 0.07
        elif gpuType == "Tesla V100-SXM2-16GB":
            maxQps = 0
            medianLatency = 0.05
        else:
            raise ValueError("Unrecognized GPU Type" + gpuType)

        settings = model.getDefaultMlPerfCfg(maxQps, medianLatency, benchConfig)

        return settings


class cutlassSgemmLoader(dataset.loader):
    checkAvailable = True

    def __init__(self, dataDir):
        self.M = 10000
        self.N = 8000
        self.K = 10000
        self.A = [0 for i in range(self.ndata)]
        self.B = [0 for i in range(self.ndata)]

    @property
    def ndata(self):
        return 20

    def preLoad(self, idxs):
        for idx in idxs:
            rng = np.random.default_rng(idx)
            a = rng.random((self.M, self.K), dtype=np.float32)
            b = rng.random((self.K, self.N), dtype=np.float32)

            a = np.asfortranarray(a)
            b = np.asfortranarray(b)

            self.A[idx] = a
            self.B[idx] = b

    def unLoad(self, idxs):
        self.A = [0 for i in range(self.ndata)]
        self.B = [0 for i in range(self.ndata)]

    def get(self, idx):
        return (self.A[idx], self.B[idx])

    def check(self, result, idx):
        checker = np.asfortranarray(np.array(result).view('<f4'))
        checker = checker.reshape(self.M, self.N, order='F')
        true_value = np.matmul(self.A[idx], self.B[idx])
        return np.allclose(checker, true_value)
