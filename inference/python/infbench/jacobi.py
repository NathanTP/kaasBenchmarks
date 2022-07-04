from . import model
from . import dataset
import numpy as np
import pycuda.driver as cuda


ROWS_PER_CTA = 8
N = 512
iters = 3000


def loadKerns(modelDir):
    mod = cuda.module_from_file(str(modelDir / "jacobi.ptx"))
    jacobiKern = mod.get_function("JacobiMethod")
    jacobiKern.prepare("iPPPPP")

    return jacobiKern


class jacobiBase(model.Model):
    nOutPost = None
    noPost = True
    nOutPre = None
    noPre = True
    preMap = model.inputMap(inp=(0, 1))
    runMap = model.inputMap(pre=(0, 1))
    postMap = model.inputMap(run=(0, 1))
    nOutRun = 2
    nConst = 0

    @staticmethod
    def pre(args):
        raise RuntimeError("Jacobi has no preprocessing step")

    @staticmethod
    def post(args):
        raise RuntimeError("Jacobi has no postprocessing step")

    @staticmethod
    def getConstants(modelDir):
        return None


class jacobi(jacobiBase):
    def __init__(self, modelArgs, devID=None):
        self.modelDir = modelArgs
        cuda.init()
        self.cudaCtx = cuda.Device(devID).make_context()
        self.jacobiKern = loadKerns(self.modelDir)

    def run(self, dat, stats=None):
        """Run the model against input 'dat'. Dat is expected to be a bytes
       object that can be converted to numpy/tvm and passed to the model as
       input."""
        self.cudaCtx.push()

        A = np.frombuffer(dat[0], dtype=np.float32)
        A.shape = (N, N)
        b = np.frombuffer(dat[1], dtype=np.float64)
        b.shape = (N, 1)

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

        for k in range(iters):
            if k % 2 == 0:
                self.jacobiKern.prepared_call(grid, block, N, A_d, b_d, x_d, x_new_d, d_d, shared_size=8*N)
            else:
                self.jacobiKern.prepared_call(grid, block, N, A_d, b_d, x_new_d, x_d, d_d, shared_size=8*N)

        if iters % 2 == 0:
            cuda.memcpy_dtoh(x_new, x_new_d)
        else:
            cuda.memcpy_dtoh(x, x_d)

        cuda.memcpy_dtoh(d, d_d)

        A_d.free()
        b_d.free()
        x_d.free()
        x_new_d.free()
        d_d.free()

        self.cudaCtx.pop()
        return (x_new, d)

    def shutdown(self):
        self.cudaCtx.detach()


class jacobiKaas(jacobiBase, model.kaasModel):
    pass


class jacobiLoader(dataset.loader):
    checkAvailable = False

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
        return [self.A.data, self.b.data]

    def check(self, result, idx):
        # print(np.frombuffer(result[0], dtype=np.float64))
        # print(struct.unpack('d', result[0]))
        # print(result[0].view('f8'))
        # print(result[1].view('f8'))
        return True
