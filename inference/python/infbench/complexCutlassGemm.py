from . import model
from . import dataset
import numpy as np
import yaml
import ctypes as ct
import pycuda.driver as cuda
import pickle


# define complex ctype as a python class
class complex(ct.Structure):
    _fields_ = [('real', ct.c_float), ('imag', ct.c_float)]

c_complex_p = ct.POINTER(complex)

#TODO CHANGE ALL OF THIS
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
alpha = 1
beta = 0

class sgemmBase(model.Model):
    noPost = True
    preMap = model.inputMap(inp=(0,))
    runMap = model.inputMap(pre=(0,), const=(0, 1, 2))
    postMap = model.inputMap(run=(0,))
    nOutRun = 1
    nOutPre = 1
    nOutPost = 1
    nConst = 3



    @staticmethod
    def pre(imgBuf):
        return imgBuf

    @staticmethod
    def post(label):
        raise AttributeError("cutlass sgemm has no post-processing")

    @staticmethod
    def getConstants(modelDir):
        #constsDir = modelDir / "cutlassSgemm_params.pkl"
        #consts = pickle.load(open(constsDir, "rb"))
        rng = np.random.default_rng(0)
        b = rng.random((K, N), dtype=np.float32) + rng.random((K, N), dtype=np.float32) * (1j)
        #b = rng.random((K, N), dtype=np.float32)
        #d = rng.random((N, 1), dtype=np.float32)
        d = rng.random((N, 1), dtype=np.float32) + rng.random((N, 1), dtype=np.float32) * (1j)
        P = rng.random((N, N), dtype=np.float32) + rng.random((N, N), dtype=np.float32) * (1j)

        return [b, d, P]
        #return [np.asfortranarray(consts[0]), np.asfortranarray(consts[1])]

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


        a = dat[3]
        b = dat[0]
        d = dat[1]
        P = dat[2]

        c = np.zeros(shape=(M, N), dtype=np.float32) + np.zeros(shape=(M, N), dtype=np.float32) * (1j)

        a_d = cuda.mem_alloc(a.nbytes)
        cuda.memcpy_htod(a_d, a)

        b_d = cuda.mem_alloc(b.nbytes)
        cuda.memcpy_htod(b_d, b)

        c_d = cuda.mem_alloc(c.nbytes)
        cuda.memset_d8(c_d, 0, c.nbytes)

        cfg = getDims(M, N, K).contents
        grid = (cfg.gridX, cfg.gridY, cfg.gridZ)
        block = (cfg.blockX, cfg.blockY, cfg.blockZ)

        params = getArg(M, N, K, alpha,
                    ct.cast(int(a_d), c_complex_p), lda,
                    ct.cast(int(b_d), c_complex_p), ldb,
                    beta,
                    ct.cast(int(c_d), c_complex_p), ldc)

        #C = A * B
        cutlassKern.prepared_call(grid, block, params.contents, shared_size=cfg.smem_size)



        lda = M
        ldb = N
        ldc = 1


        #C' = C * K

        d = dat[1]
        e = np.zeros(shape=(M, 1), dtype=np.float32) + np.zeros(shape=(M, 1), dtype=np.float32) * (1j)

        d_d = cuda.mem_alloc(d.nbytes)
        cuda.memcpy_htod(d_d, d)

        e_d = cuda.mem_alloc(e.nbytes)
        cuda.memset_d8(e_d, 0, e.nbytes)

        cfg = getDims(M, 1, N).contents
        grid = (cfg.gridX, cfg.gridY, cfg.gridZ)
        block = (cfg.blockX, cfg.blockY, cfg.blockZ)

        smem = cfg.smem_size
        params = getArg(M, 1, K, alpha,
                    ct.cast(int(c_d), c_complex_p), lda,
                    ct.cast(int(d_d), c_complex_p), ldb,
                    beta,
                    ct.cast(int(e_d), c_complex_p), ldc)

        #e = C * d
        cutlassKern.prepared_call(grid, block, params.contents, shared_size=smem)

        cuda.Context.synchronize()
        cuda.memcpy_dtoh(e, e_d)
        #print(e)

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
        rng = np.random.default_rng(0)
        #a = rng.random((M, K), dtype=np.float32)
        a = rng.random((M, K), dtype=np.float32) + rng.random((M, K), dtype=np.float32) * (1j)
        self.a = a

    def unLoad(self, idxs):
        self.a = None

    def get(self, idx):
        return [self.a]

    def check(self, result, idx):
        actual = np.asfortranarray(np.array(result).view('<c8'))
        actual = actual.reshape(M, 1, order='F')
        consts = sgemmBase.getConstants(None)
        b = consts[0]
        d = consts[1]
        expected = np.matmul(np.matmul(self.a, b), d)
        return np.allclose(actual, expected, rtol=0.5)
