from . import model
from . import dataset
import numpy as np
import ctypes as ct
import pycuda.driver as cuda


def loadAdapter(modelDir):
    libc = ct.cdll.LoadLibrary(str(modelDir / "cutlassAdapters.so"))
    getArg = libc.adaptSGEMMArgs
    c_float_p = ct.POINTER(ct.c_float)
    getArg.argtypes = [ct.c_int, ct.c_int, ct.c_int, ct.c_float, c_float_p, ct.c_int,
                       c_float_p, ct.c_int, ct.c_float, c_float_p, ct.c_int]

    # that it's a byte array of the same size (320 bytes in this case)
    getArg.restype = ct.POINTER(ct.c_byte*320)

    getDims = libc.getCudaConfig
    # M, N, K
    getDims.argtypes = [ct.c_int, ct.c_int, ct.c_int]
    getDims.restype = ct.POINTER(kernelConfig)
    return (getArg, getDims)


def loadKerns(modelDir):
    mod = cuda.module_from_file(str(modelDir / "cutlass.cubin"))

    # Since the cutlass kernel came from a template, the name is crazy long.
    # Unfortunately, extern "C" doesn't fix the issue. This string is obtained
    # by running "nm" on the cubin
    cutlassKern = mod.get_function("_ZN7cutlass6KernelINS_4gemm6kernel4GemmINS1_11threadblock12MmaPipelinedINS1_9GemmShapeILi128ELi128ELi8EEENS_9transform11threadblock22PredicatedTileIteratorINS_11MatrixShapeILi128ELi8EEEfNS_6layout8RowMajorELi1ENS8_30PitchLinearStripminedThreadMapINSD_16PitchLinearShapeILi8ELi128EEELi256ELi1EEELi1EEENS9_19RegularTileIteratorISC_fNSD_11ColumnMajorELi1ENS8_33TransposePitchLinearThreadMapSimtISI_EELi4EEENSA_INSB_ILi8ELi128EEEfSE_Li0ENSF_INSG_ILi128ELi8EEELi256ELi1EEELi1EEENSK_ISP_fSE_Li0ESR_Li4EEEfSE_NS4_9MmaPolicyINS1_4warp7MmaSimtINS6_ILi32ELi64ELi8EEEfSL_fSE_fSE_NSV_13MmaSimtPolicyINSB_ILi4ELi8EEENSD_19RowMajorInterleavedILi2EEENS6_ILi4ELi4ELi1EEEEELi1ELNS_16ComplexTransformE0ELS14_0EbEENSB_ILi4ELi0EEENSB_ILi0ELi0EEELi1EEENS_21NumericArrayConverterIffLi4ELNS_15FloatRoundStyleE2EEES1B_bEENS_8epilogue11threadblock8EpilogueIS7_S15_Li1ENS1E_22PredicatedTileIteratorINS1E_26OutputTileOptimalThreadMapINS1E_15OutputTileShapeILi128ELi1ELi4ELi4ELi1EEENS1I_ILi1ELi4ELi2ELi1ELi8EEELi256ELi1ELi32EEEfEENS1D_4warp20FragmentIteratorSimtISX_NS1_6thread3MmaINS6_ILi8ELi8ELi1EEEfSL_fSE_fSE_NS_4arch13OpMultiplyAddEbEESE_S13_EENS1N_16TileIteratorSimtISX_S1U_fSE_S13_EENS1E_18SharedLoadIteratorINS1L_18CompactedThreadMapEfLi4EEENS1D_6thread17LinearCombinationIfLi1EffLNS21_9ScaleType4KindE0ELS1A_2EEENSB_ILi0ELi17EEELi1EEENS4_30GemmIdentityThreadblockSwizzleILi1EEELb0EEEEEvNT_6ParamsE")

    # The kernel takes a Params struct as an argument (by value). Rather than
    # try to define that struct in python, we instead find its size manually
    # (in cuda using sizeof()) and then specify a byte array of the same size
    # here. Pycuda doesn't care about the type in practice, it only needs the
    # size. This type string is defined by python's "struct" module.
    cutlassKern.prepare("320s")

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


redDim = 2
M = 100
N = 8000
K = 10000
alpha = 1
beta = 1


class sgemmBase(model.Model):
    noPost = True
    preMap = model.inputMap(inp=(0))
    runMap = model.inputMap(pre=(0), const=(0, 1))
    postMap = model.inputMap(run=(0,))
    nOutRun = 1
    nOutPre = 1
    nOutPost = 1
    nConst = 2

    @staticmethod
    def pre(imgBuf):
        return imgBuf

    @staticmethod
    def post(label):
        raise AttributeError("cutlass sgemm has no post-processing")

    @staticmethod
    def getConstants(modelDir):
        rng = np.random.default_rng(0)
        b = np.asfortranarray(rng.random((K, N), dtype=np.float32))
        d = np.asfortranarray(rng.random((N, redDim), dtype=np.float32))

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

    def run(self, dat, stats=None):
        """Run the model against input 'dat'. Dat is expected to be a bytes
       object that can be converted to numpy/tvm and passed to the model as
       input."""
        import pycuda.autoinit  # NOQA

        lda = M
        ldb = K
        ldc = M

        getArg, getDims = loadAdapter(self.modelDir.parent)
        refKern, cutlassKern = loadKerns(self.modelDir.parent)

        a = dat[2]
        b = dat[0]
        d = dat[1]
        aSz = M*K*4
        bSz = K*N*4
        cSz = M*N*4
        dSz = N*redDim*4
        eSz = M*redDim*4

        a_d = cuda.mem_alloc(aSz)
        cuda.memcpy_htod(a_d, a)

        b_d = cuda.mem_alloc(bSz)
        cuda.memcpy_htod(b_d, b)

        c_d = cuda.mem_alloc(cSz)
        cuda.memset_d8(c_d, 0, cSz)

        cfg = getDims(M, N, K).contents
        grid = (cfg.gridX, cfg.gridY, cfg.gridZ)
        block = (cfg.blockX, cfg.blockY, cfg.blockZ)

        params = getArg(M, N, K, alpha,
                        ct.cast(int(a_d), ct.POINTER(ct.c_float)), lda,
                        ct.cast(int(b_d), ct.POINTER(ct.c_float)), ldb,
                        beta,
                        ct.cast(int(c_d), ct.POINTER(ct.c_float)), ldc)

        cutlassKern.prepared_call(grid, block, params.contents, shared_size=cfg.smem_size)

        lda = M
        ldb = N
        ldc = M

        d_d = cuda.mem_alloc(dSz)
        cuda.memcpy_htod(d_d, d)

        e_d = cuda.mem_alloc(eSz)
        cuda.memset_d8(e_d, 0, eSz)

        cfg = getDims(M, redDim, N).contents
        grid = (cfg.gridX, cfg.gridY, cfg.gridZ)
        block = (cfg.blockX, cfg.blockY, cfg.blockZ)

        smem = cfg.smem_size
        params = getArg(M, redDim, N, alpha,
                        ct.cast(int(c_d), ct.POINTER(ct.c_float)), lda,
                        ct.cast(int(d_d), ct.POINTER(ct.c_float)), ldb,
                        beta,
                        ct.cast(int(e_d), ct.POINTER(ct.c_float)), ldc)

        cutlassKern.prepared_call(grid, block, params.contents, shared_size=smem)

        cuda.Context.synchronize()

        e = bytearray(eSz)
        cuda.memcpy_dtoh(e, e_d)

        return e


class sgemmKaas(sgemmBase, model.kaasModel):
    pass


class cutlassSgemmLoader(dataset.loader):
    checkAvailable = True

    def __init__(self, dataDir):
        pass

    @property
    def ndata(self):
        return 1

    def preLoad(self, idxs):
        # rng = np.random.default_rng(0)
        # a = rng.random((M, K), dtype=np.float32)
        # self.a = np.asfortranarray(a)

        aTile = np.arange(1, 101, dtype=np.float32) / 100
        self.a = np.asfortranarray(np.tile(aTile, (M, int(K / 100))))

    def unLoad(self, idxs):
        self.a = None

    def get(self, idx):
        return [self.a.ravel(order='K').data]

    def check(self, result, idx):
        actual = np.ndarray(shape=(M, redDim), buffer=result[0], order='F', dtype=np.float32)

        a = self.a

        consts = sgemmBase.getConstants(None)
        b = np.ndarray(shape=(K, N), buffer=consts[0], order='F', dtype=np.float32)
        d = np.ndarray(shape=(N, redDim), buffer=consts[1], order='F', dtype=np.float32)

        expected = np.matmul(np.matmul(a, b), d)
        # print(expected[:][:10])
        # print(actual[:][:10])
        # print("Close?: ", np.allclose(actual, expected, rtol=0.05))

        return np.allclose(actual, expected, rtol=0.05)
