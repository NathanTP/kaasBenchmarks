import kaas
import ctypes as ct

redDim = 2
M = 100
N = 8000
K = 10000
alpha = 1
beta = 1


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
    getDims = libc.getCudaConfig
    # M, N, K
    getDims.argtypes = [ct.c_int, ct.c_int, ct.c_int]
    getDims.restype = ct.POINTER(kernelConfig)

    return getDims


def createReq(M, N, K, alpha, beta):
    lda = M
    ldb = K
    ldc = M

    getDims = loadDims()

    aBuf = kaas.bufferSpec('a', M*K*4, ephemeral=False)
    bBuf = kaas.bufferSpec('b', K*N*4, ephemeral=False)
    cBuf = kaas.bufferSpec('c', M*N*4, ephemeral=True)

    dBuf = kaas.bufferSpec('d', N*redDim*4, ephemeral=False)
    eBuf = kaas.bufferSpec('e', M*redDim*4, ephemeral=False)

    literals = [kaas.literalSpec('f', alpha), kaas.literalSpec('f', beta),
                kaas.literalSpec('i', M), kaas.literalSpec('i', N),
                kaas.literalSpec('i', K), kaas.literalSpec('i', lda),
                kaas.literalSpec('i', ldb), kaas.literalSpec('i', ldc)]

    cfg = getDims(M, N, K).contents
    grid = (cfg.gridX, cfg.gridY, cfg.gridZ)
    block = (cfg.blockX, cfg.blockY, cfg.blockZ)
    smem = cfg.smem_size
    firstKern = kaas.kernelSpec(kaas.builtins["cutlass"], "sgemm0",
                                grid, block, sharedSize=smem,
                                arguments=[(aBuf, 'i'), (bBuf, 'i'), (cBuf, 'o')],
                                literals=literals)

    literals = [kaas.literalSpec('f', alpha), kaas.literalSpec('f', beta),
                kaas.literalSpec('i', M), kaas.literalSpec('i', redDim),
                kaas.literalSpec('i', N), kaas.literalSpec('i', M),
                kaas.literalSpec('i', N), kaas.literalSpec('i', M)]

    cfg = getDims(M, redDim, N).contents
    grid = (cfg.gridX, cfg.gridY, cfg.gridZ)
    block = (cfg.blockX, cfg.blockY, cfg.blockZ)
    smem = cfg.smem_size
    redKern = kaas.kernelSpec(kaas.builtins["cutlass"], "sgemm0",
                              grid, block, sharedSize=smem,
                              arguments=[(cBuf, 'i'), (dBuf, 'i'), (eBuf, 'o')],
                              literals=literals)

    req = kaas.kaasReq([firstKern, redKern])
    return req
