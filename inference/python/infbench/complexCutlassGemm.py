from . import model
from . import dataset
from . import util
import numpy as np
import ctypes as ct
import pycuda.driver as cuda
import pycuda.tools
import pickle


# define complex ctype as a python class
class complex(ct.Structure):
    _fields_ = [('real', ct.c_float), ('imag', ct.c_float)]


c_complex_p = ct.POINTER(complex)


def loadAdapter(modelDir):
    libc = ct.cdll.LoadLibrary(str(modelDir / "./cutlassAdapters.so"))
    getArg = libc.adaptSGEMMArgs
    getArg.argtypes = [ct.c_int, ct.c_int, ct.c_int, ct.c_float, c_complex_p, ct.c_int,
                       c_complex_p, ct.c_int, ct.c_float, c_complex_p, ct.c_int]
    # Instead of trying to define the Params struct in python, we just pretend
    # that it's a byte array of the same size (320 bytes in this case)
    getArg.restype = ct.POINTER(ct.c_byte*328)

    getDims = libc.getCudaConfig
    # M, N, K
    getDims.argtypes = [ct.c_int, ct.c_int, ct.c_int]
    getDims.restype = ct.POINTER(kernelConfig)

    return (getArg, getDims)


def loadKerns(modelDir):
    mod = cuda.module_from_file(str(modelDir / "./cutlass.cubin"))

    # Since the cutlass kernel came from a template, the name is crazy long.
    # Unfortunately, extern "C" doesn't fix the issue. This string is obtained
    # by running "nm" on the cubin
    cutlassKern = mod.get_function("_ZN7cutlass6KernelINS_4gemm6kernel4GemmINS1_11threadblock12MmaPipelinedINS1_9GemmShapeILi128ELi128ELi8EEENS_9transform11threadblock22PredicatedTileIteratorINS_11MatrixShapeILi128ELi8EEENS_7complexIfEENS_6layout8RowMajorELi1ENS8_30PitchLinearStripminedThreadMapINSF_16PitchLinearShapeILi8ELi128EEELi256ELi1EEELi1EEENS9_19RegularTileIteratorISC_SE_NSF_11ColumnMajorELi1ENS8_33TransposePitchLinearThreadMapSimtISK_EELi8EEENSA_INSB_ILi8ELi128EEESE_SG_Li0ENSH_INSI_ILi128ELi8EEELi256ELi1EEELi1EEENSM_ISR_SE_SG_Li0EST_Li8EEESE_SG_NS4_9MmaPolicyINS1_4warp7MmaSimtINS6_ILi32ELi64ELi8EEESE_SN_SE_SG_SE_SG_NSX_13MmaSimtPolicyINSB_ILi4ELi8EEENSF_19RowMajorInterleavedILi2EEENS6_ILi2ELi2ELi1EEEEELi1ELNS_16ComplexTransformE0ELS16_0EbEENSB_ILi2ELi0EEENSB_ILi0ELi0EEELi1EEENS_21NumericArrayConverterISE_SE_Li4ELNS_15FloatRoundStyleE2EEES1D_bEENS_8epilogue11threadblock8EpilogueIS7_S17_Li1ENS1G_22PredicatedTileIteratorINS1G_26OutputTileOptimalThreadMapINS1G_15OutputTileShapeILi128ELi1ELi4ELi4ELi1EEENS1K_ILi1ELi2ELi4ELi1ELi8EEELi256ELi1ELi64EEESE_EENS1F_4warp20FragmentIteratorSimtISZ_NS1_6thread3MmaINS6_ILi8ELi8ELi1EEESE_SN_SE_SG_SE_SG_NS_4arch13OpMultiplyAddEbEESG_S15_EENS1P_16TileIteratorSimtISZ_S1W_SE_SG_S15_EENS1G_18SharedLoadIteratorINS1N_18CompactedThreadMapESE_Li8EEENS1F_6thread17LinearCombinationISE_Li1ESE_SE_LNS23_9ScaleType4KindE0ELS1C_2EEENSB_ILi0ELi9EEELi1EEENS4_30GemmIdentityThreadblockSwizzleILi1EEELb0EEEEEvNT_6ParamsE")

    # The kernel takes a Params struct as an argument (by value). Rather than
    # try to define that struct in python, we instead find its size manually
    # (in cuda using sizeof()) and then specify a byte array of the same size
    # here. Pycuda doesn't care about the type in practice, it only needs the
    # size. This type string is defined by python's "struct" module.
    cutlassKern.prepare("328s")

    refKern = mod.get_function("ReferenceGemm_kernel")
    # See python's struct module for a description of this type string
    refKern.prepare("iiifPiPifPi")

    return refKern, cutlassKern


class kernelConfig(ct.Structure):
    """This mirrors the CudaConfig struct defined in cutlassAdapters.h"""
    _fields_ = [
        ("gridX", ct.c_int),
        ("gridY", ct.c_int),
        ("gridZ", ct.c_int),
        ("blockX", ct.c_int),
        ("blockY", ct.c_int),
        ("blockZ", ct.c_int),
        ("smem_size", ct.c_int)
    ]


M = 100
N = 25000
K = 10000
redDim = 1
alpha = 1
beta = 0


class sgemmBase(model.Model):
    noPost = True
    preMap = model.inputMap(inp=(0,))
    runMap = model.inputMap(pre=(0,), const=(0, 1))
    postMap = model.inputMap()
    nOutRun = 1
    nOutPre = 1
    nOutPost = None
    nConst = 2

    @staticmethod
    def pre(imgBuf):
        return imgBuf

    @staticmethod
    def post(label):
        raise AttributeError("cutlass sgemm has no post-processing")

    @staticmethod
    def getConstants(modelDir):
        with open(modelDir / 'complexCutlassGemm_consts.pkl', 'rb') as f:
            b, d = pickle.load(f)

        return [b.ravel(order='K').data, d.ravel(order='K').data]

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
    def __init__(self, modelArgs):
        self.modelDir = modelArgs
        self.initialized = False
        self.dbufA = None
        self.dbufB = None
        self.dbufD = None
        self.dbufC = None
        self.dbufE = None

        cuda.init()
        self.cudaCtx = pycuda.tools.make_default_context()
        util.cudaProfilerResetCtx()

    def __del__(self):
        self.cudaCtx.detach()

    def run(self, dat, stats=None):
        """Run the model against input 'dat'. Dat is expected to be a bytes
       object that can be converted to numpy/tvm and passed to the model as
       input."""

        lda = M
        ldb = K
        ldc = M

        getArg, getDims = loadAdapter(self.modelDir.parent)
        refKern, cutlassKern = loadKerns(self.modelDir.parent)

        a = dat[2]
        b = dat[0]
        d = dat[1]
        aSz = M*K*8
        bSz = K*N*8
        cSz = M*N*8
        dSz = N*redDim*8
        eSz = M*redDim*8

        if self.dbufA is None:
            self.dbufA = cuda.mem_alloc(aSz)
            self.dbufB = cuda.mem_alloc(bSz)
            self.dbufC = cuda.mem_alloc(cSz)
            self.dbufD = cuda.mem_alloc(dSz)
            self.dbufE = cuda.mem_alloc(eSz)

        if not self.initialized:
            cuda.memcpy_htod(self.dbufB, b)
            cuda.memcpy_htod(self.dbufD, d)
            self.initialized = True

        cuda.memcpy_htod(self.dbufA, a)
        cuda.memset_d8(self.dbufC, 0, cSz)
        cuda.memset_d8(self.dbufE, 0, eSz)

        cfg = getDims(M, N, K).contents
        grid = (cfg.gridX, cfg.gridY, cfg.gridZ)
        block = (cfg.blockX, cfg.blockY, cfg.blockZ)

        params = getArg(M, N, K, alpha,
                        ct.cast(int(self.dbufA), c_complex_p), lda,
                        ct.cast(int(self.dbufB), c_complex_p), ldb,
                        beta,
                        ct.cast(int(self.dbufC), c_complex_p), ldc)

        cutlassKern.prepared_call(grid, block, params.contents, shared_size=cfg.smem_size)

        lda = M
        ldb = N
        ldc = M

        cfg = getDims(M, redDim, N).contents
        grid = (cfg.gridX, cfg.gridY, cfg.gridZ)
        block = (cfg.blockX, cfg.blockY, cfg.blockZ)

        smem = cfg.smem_size
        params = getArg(M, redDim, N, alpha,
                        ct.cast(int(self.dbufC), c_complex_p), lda,
                        ct.cast(int(self.dbufD), c_complex_p), ldb,
                        beta,
                        ct.cast(int(self.dbufE), c_complex_p), ldc)

        cutlassKern.prepared_call(grid, block, params.contents, shared_size=smem)

        cuda.Context.synchronize()

        e = bytearray(eSz)

        cuda.memcpy_dtoh(e, self.dbufE)

        return (e,)


class sgemmKaas(sgemmBase, model.kaasModel):

    @staticmethod
    def getConstants(modelDir):
        """Default constant loader assumes the kaasModel simply pickled their
        constants and we can load them directly."""
        constants = sgemmBase.getConstants(modelDir)
        return constants


class cutlassSgemmLoader(dataset.loader):
    checkAvailable = True

    def __init__(self, dataDir):
        self.dataDir = dataDir
        pass

    @property
    def ndata(self):
        return 1

    def preLoad(self, idxs):
        with open(self.dataDir / 'complexCutlassGemm_input.pkl', 'rb') as f:
            a = pickle.load(f)

        self.a = a

    def unLoad(self, idxs):
        self.a = None

    def get(self, idx):
        return [self.a.ravel(order='K').data]

    def check(self, result, idx):
        actual = np.ndarray(shape=(M, redDim), buffer=result[0], order='F', dtype=np.csingle)
        consts = sgemmBase.getConstants(self.dataDir)
        b = np.ndarray(shape=(K, N), buffer=consts[0], order='F', dtype=np.csingle)
        d = np.ndarray(shape=(N, redDim), buffer=consts[1], order='F', dtype=np.csingle)
        expected = np.matmul(np.matmul(self.a, b), d)
        # print(actual)
        # print(expected)
        return np.allclose(actual, expected, rtol=0.5)
