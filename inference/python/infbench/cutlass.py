from . import model
from . import dataset
import numpy as np
import yaml
import ctypes as ct
import pycuda.driver as cuda
import pickle


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


class sgemmBase(model.Model):
    noPost = True
    preMap = model.inputMap(inp=(0))
    runMap = model.inputMap(const=(0, 1), pre=(0))
    postMap = model.inputMap(run=(0))
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
        constsDir = modelDir / "cutlassSgemm_params.pkl"
        return pickle.load(open(constsDir, "rb"))


class cutlassSgemm(sgemmBase):
    def __init__(self, modelArgs):
        self.M = 1000
        self.N = 10000
        self.K = 800
        self.alpha = 1
        self.beta = 0
        self.modelDir = modelArgs

    def run(self, dat, stats=None):
        """Run the model against input 'dat'. Dat is expected to be a bytes
       object that can be converted to numpy/tvm and passed to the model as
       input."""
        import pycuda.autoinit  # NOQA

        lda = self.M
        ldb = self.K
        ldc = self.M

        getArg, getDims = loadAdapter(self.modelDir.parent)
        refKern, cutlassKern = loadKerns(self.modelDir.parent)

        #print(len(dat))
        #print(dat[0].shape)
        #print(dat[1].shape)
        #print(dat[2].shape)

        a = dat[2]
        b = dat[0]
        c = np.zeros(shape=(self.M, self.N), order='F', dtype=np.float32)

        a_d = cuda.mem_alloc(a.nbytes)
        cuda.memcpy_htod(a_d, a)

        b_d = cuda.mem_alloc(b.nbytes)
        cuda.memcpy_htod(b_d, b)

        c_d = cuda.mem_alloc(c.nbytes)
        cuda.memset_d8(c_d, 0, c.nbytes)

        cfg = getDims(self.M, self.N, self.K).contents
        grid = (cfg.gridX, cfg.gridY, cfg.gridZ)
        block = (cfg.blockX, cfg.blockY, cfg.blockZ)

        params = getArg(self.M, self.N, self.K, self.alpha,
                        ct.cast(int(a_d), ct.POINTER(ct.c_float)), lda,
                        ct.cast(int(b_d), ct.POINTER(ct.c_float)), ldb,
                        self.beta,
                        ct.cast(int(c_d), ct.POINTER(ct.c_float)), ldc)

        cutlassKern.prepared_call(grid, block, params.contents, shared_size=cfg.smem_size)
        #cuda.Context.synchronize()

        lda = self.M
        ldb = self.N
        ldc = 1

        print("hello")

        d = dat[1]
        e = np.zeros(shape=(self.M, 1), order='F', dtype=np.float32)

        d_d = cuda.mem_alloc(d.nbytes)
        cuda.memcpy_htod(d_d, d)

        e_d = cuda.mem_alloc(e.nbytes)
        cuda.memset_d8(e_d, 0, e.nbytes)

        cfg = getDims(self.M, 1, self.N).contents
        grid = (cfg.gridX, cfg.gridY, cfg.gridZ)
        block = (cfg.blockX, cfg.blockY, cfg.blockZ)

        smem = cfg.smem_size
        params = getArg(self.M, 1, self.N, self.alpha,
                        ct.cast(int(c_d), ct.POINTER(ct.c_float)), lda,
                        ct.cast(int(d_d), ct.POINTER(ct.c_float)), ldb,
                        self.beta,
                        ct.cast(int(e_d), ct.POINTER(ct.c_float)), ldc)


        #literals = [kaas.literalSpec('f', alpha), kaas.literalSpec('f', beta), kaas.literalSpec('f', M), kaas.literalSpec('f', 1), kaas.literalSpec('f', N), kaas.literalSpec('f', M), kaas.literalSpec('f', N), kaas.literalSpec('f', M)]
        #secondKern = kaas.kernelSpec(kaas.builtins["cutlass"], "sgemm0", grid, block, sharedSize=smem, arguments=[(cBuf, 'i'), (dBuf, 'i'), (eBuf, 'o')], literals=literals)
        cutlassKern.prepared_call(grid, block, params.contents, shared_size=smem)
        cuda.Context.synchronize()



        cuda.memcpy_dtoh(e, e_d)

        a_d.free()
        b_d.free()
        c_d.free()
        d_d.free()
        e_d.free()

        return e

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
        self.dataDir = dataDir
        self.M = 1000
        self.N = 10000
        self.K = 800
        nPoints = 20
        self.A = [0 for i in range(nPoints)]
        self.B = None
        self.d = None #[0 for i in range(nPoints)]

    @property
    def ndata(self):
        return 20

    def preLoad(self, idxs):
        thing = pickle.load(open(self.dataDir / "cutlassSgemm_params.pkl", "rb"))
        self.B = thing[0]
        self.d = thing[1]
        for idx in idxs:
            rng = np.random.default_rng(idx)
            a = rng.random((self.M, self.K), dtype=np.float32)
            #b = rng.random((self.K, self.N), dtype=np.float32)
            d = rng.random((self.N, 1), dtype=np.float32)

            a = np.asfortranarray(a)
            d = np.asfortranarray(d)

            self.A[idx] = a
            #self.d[idx] = d

    def unLoad(self, idxs):
        self.A = [0 for i in range(self.ndata)]
        self.B = [0 for i in range(self.ndata)]

    def get(self, idx):
        return (self.A[idx],)
        #return (self.A[idx], self.d[idx])

    def check(self, result, idx):
        checker = np.asfortranarray(np.array(result).view('<f4'))
        print(checker.shape)
        checker = checker.reshape(self.M, order='F')
        true_value = np.matmul(self.A[idx], self.B)
        #print(true_value)
        print(self.d.shape)
        true_value = np.matmul(true_value, self.d)
        print(true_value)
        print(checker)
        #print(checker)
        #print(true_value)
        #return False
        return np.allclose(checker, true_value, rtol=20)
        #return False
