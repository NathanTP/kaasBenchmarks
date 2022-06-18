import kaas
import ctypes as ct
import numpy as np


# define complex ctype as a python class
class complex(ct.Structure):
    _fields_ = [('real', ct.c_float), ('imag', ct.c_float)]


c_complex_p = ct.POINTER(complex)


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


def loadDims():
    libc = ct.cdll.LoadLibrary("./getDims.so")
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

    return getDims


def createReq(M, N, K, redDim, alpha, beta):
    dtypeSize = 8

    lda = M
    ldb = K
    ldc = M

    getDims = loadDims()
    cfg = getDims(M, N, K).contents
    grid = (cfg.gridX, cfg.gridY, cfg.gridZ)
    block = (cfg.blockX, cfg.blockY, cfg.blockZ)

    smem = cfg.smem_size

    aBuf = kaas.bufferSpec('a', M*K*dtypeSize, ephemeral=False)

    bBuf = kaas.bufferSpec('b', K*N*dtypeSize, ephemeral=False)

    cBuf = kaas.bufferSpec('c', M*N*dtypeSize, ephemeral=True)
    literals = [kaas.literalSpec('f', alpha), kaas.literalSpec('f', beta),
                kaas.literalSpec('f', M), kaas.literalSpec('f', N),
                kaas.literalSpec('f', K), kaas.literalSpec('f', lda),
                kaas.literalSpec('f', ldb), kaas.literalSpec('f', ldc)]

    firstKern = kaas.kernelSpec(kaas.builtins["complexCutlass"], "complexGemm0",
                                grid, block, sharedSize=smem,
                                arguments=[(aBuf, 'i'), (bBuf, 'i'), (cBuf, 't')],
                                literals=literals)

    dBuf = kaas.bufferSpec('d', N*redDim*dtypeSize, ephemeral=False)
    eBuf = kaas.bufferSpec('e', M*redDim*dtypeSize, ephemeral=True)

    cfg = getDims(M, redDim, N).contents
    grid = (cfg.gridX, cfg.gridY, cfg.gridZ)
    block = (cfg.blockX, cfg.blockY, cfg.blockZ)

    smem = cfg.smem_size

    literals = [kaas.literalSpec('f', alpha), kaas.literalSpec('f', beta),
                kaas.literalSpec('f', M), kaas.literalSpec('f', 1),
                kaas.literalSpec('f', N), kaas.literalSpec('f', M),
                kaas.literalSpec('f', N), kaas.literalSpec('f', M)]

    secondKern = kaas.kernelSpec(kaas.builtins["complexCutlass"], "complexGemm0",
                                 grid, block, sharedSize=smem,
                                 arguments=[(cBuf, 'i'), (dBuf, 'i'), (eBuf, 'o')],
                                 literals=literals)

    req = kaas.kaasReq([firstKern, secondKern])
    return req


def generateData(M, N, K, redDim):
    rng = np.random.default_rng(0)

    a = np.asfortranarray(rng.standard_normal((M, K), dtype=np.float32)) + \
        np.asfortranarray(rng.standard_normal((M, K), dtype=np.float32) * (1j))

    b = np.asfortranarray(rng.standard_normal((K, N), dtype=np.float32)) + \
        np.asfortranarray(rng.standard_normal((K, N), dtype=np.float32) * (1j))

    d = np.asfortranarray(rng.standard_normal((N, redDim), dtype=np.float32)) + \
        np.asfortranarray(rng.standard_normal((N, redDim), dtype=np.float32) * (1j))

    return (a, (b, d))
