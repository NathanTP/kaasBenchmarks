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

    @staticmethod
    def getPerfEstimates(gpuType):
        if gpuType == "Tesla K20c":
            maxQps = 0
            medianLatency = 0.07
        elif gpuType == "Tesla V100-SXM2-16GB":
            maxQps = 0
            medianLatency = 0.05
        else:
            raise ValueError("Unrecoginzied GPU Type" + gpuType)

        return maxQps, medianLatency

    @classmethod
    def getMlPerfCfg(cls, gpuType, benchConfig):
        maxQps, medianLatency = cls.getPerfEstimates(gpuType)
        return model.getDefaultMlPerfCfg(maxQps, medianLatency, benchConfig)


class sgemm(sgemmBase):
    def run(self, dat):
        """Run the model against input 'dat'. Dat is expected to be a bytes
       object that can be converted to numpy/tvm and passed to the model as
       input."""
        pass


class sgemmKaas(sgemmBase, model.kaasModel):
    pass


class cutlassSgemmLoader(dataset.loader):
    checkAvailable = True

    def __init__(self, dataDir):
        self.M = 10000
        self.N = 8000
        self.K = 10000

    @property
    def ndata(self):
        return 1000

    def preLoad(self, idxs):
        pass

    def unLoad(self, idxs):
        pass

    def get(self, idx):
        rng = np.random.default_rng(0)
        a = rng.random((self.M, self.K), dtype=np.float32)
        b = rng.random((self.K, self.N), dtype=np.float32)
        #b = np.arange(self.K * self.N, dtype=np.float32)
        #self.checkA = np.reshape(a, (self.M, self.K))
        self.checkA = a
        self.checkB = np.reshape(b, (self.K, self.N))
        #b = np.reshape(b, (self.K, self.N))
        #a = np.reshape(a, (self.M, self.K), order='F')
        b = np.reshape(b, (self.K, self.N), order='F')
        self.a = a
        self.b = b
        self.a = np.asfortranarray(a)
        self.b = np.asfortranarray(b)
        return (self.a, self.b)

    def check(self, result, idx):
        checker = np.asfortranarray(np.array(result).view('<f4'))
        checker = checker.reshape(self.M, self.N, order='F')
        #temp = np.array(result)
        #print(temp.dtype)
        #temp = temp.tobytes()
        print(checker)
        #print(np.frombuffer(temp, dtype=np.float32))
        thing = np.matmul(self.checkA, self.checkB)
        print(thing)
        print(thing.shape)
        #print(checker - thing)
        return True

