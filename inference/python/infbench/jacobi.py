from . import model
from . import dataset
import numpy as np
import pycuda.driver as cuda



ROWS_PER_CTA = 8
N = 512
iters = 3000

def loadKerns(modelDir):
    print(str(modelDir / "jacobi.ptx"))
    mod = cuda.module_from_file(str( modelDir / "jacobi.ptx"))
    jacobiKern = mod.get_function("JacobiMethod")
    jacobiKern.prepare("iPPPPP")

    return jacobiKern

class jacobiBase(model.Model):
    noPost = True
    preMap = model.inputMap(inp=(0, 1))
    runMap = model.inputMap(pre=(0, 1))
    postMap = model.inputMap(run=(0,))
    nOutRun = 2
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
        return None

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

class jacobi(jacobiBase):
    def __init__(self, modelArgs):
        self.modelDir = modelArgs


    def run(self, dat, stats=None):
        """Run the model against input 'dat'. Dat is expected to be a bytes
       object that can be converted to numpy/tvm and passed to the model as
       input."""
        import pycuda.autoinit #NOQA
        jacobiKern = loadKerns(self.modelDir)

        A = dat[0]
        b = dat[1]

        x = np.zeros(shape=(N, 1), dtype=np.float64)
        x_new = np.zeros(shape=(N, 1), dtype=np.float64)
        d = np.zeros(1, dtype=np.float64)

        A_d = cuda.mem_alloc(A.nbytes)
        cuda.memcpy_htod(A_d, A)

        b_d = cuda.mem_alloc(b.nbytes)
        cuda.memcpy_htod(b_d, b)

        x_d = cuda.mem_alloc(x.nbytes)
        cuda.memset_d8(x_d, 0, x.nbytes)

        x_new_d = cuda.mem_alloc(x_new.nbytes)
        cuda.memset_d8(x_new_d, 0, x_new.nbytes)

        d_d = cuda.mem_alloc(d.nbytes)
        cuda.memset_d8(d_d, 0, d.nbytes)

        grid = (256, 1, 1)
        block = ((N // ROWS_PER_CTA) + 2, 1, 1)

        #print("Grid is: ", grid)
        #print("Block is: ", block)

        for k in range(iters):
            if k % 2 == 0:
                jacobiKern.prepared_call(grid, block, N, A_d, b_d, x_d, x_new_d, d_d, shared_size=8*N)
            else:
                jacobiKern.prepared_call(grid, block, N, A_d, b_d, x_new_d, x_d, d_d, shared_size=8*N)

        if iters % 2 == 0:
            cuda.memcpy_dtoh(x_new, x_new_d)
            print("CUDA result is:")
            print(x_new)

        else:
            cuda.memcpy_dtoh(x, x_d)
            print("CUDA result is:")
            print(x)

        # Relative difference between numpy and cuda result
        np_res = np.linalg.solve(A, b)
        print("Diff between numpy and cuda is:")
        if iters % 2 == 0:
            print(np.abs((np_res - x_new) / np_res))
        else:
            print(np.abs((np_res - x) / np_res))

        # This should print out the error
        cuda.memcpy_dtoh(d, d_d)
        print("CUDA error is:")
        print(d)
        return [x_new, d]



class jacobiKaas(jacobiBase, model.kaasModel):
    pass


class jacobiLoader(dataset.loader):
    checkAvailable = True

    def __init__(self, dataDir):
        pass

    @property
    def ndata(self):
        return 1

    def preLoad(self, idxs):
        rng = np.random.default_rng(40)
        self.A = rng.random((N, N), dtype=np.float32)
        fill_arr = np.sum(np.abs(self.A), axis=1)
        np.fill_diagonal(self.A, fill_arr)
        self.b = rng.random((N, 1), dtype=np.float64)

    def unLoad(self, idxs):
        self.A = None
        self.b = None

    def get(self, idx):
        return [self.A, self.b]


    def check(self, result, idx):
        print(result[0])
        print(result[1])
        return True
