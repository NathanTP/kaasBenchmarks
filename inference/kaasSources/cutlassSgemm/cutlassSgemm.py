from libff import kaas
import numpy as np
import ctypes as ct


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

def createReq(M, N, K, alpha, beta, a, b, c, d, e):
    lda = M
    ldb = K
    ldc = M

    #libffCtx = getCtx(remote=False)
    getDims = loadDims()
    #rng = np.random.default_rng(0)
    #a = rng.random((M, K), dtype=np.float32)
    #b = rng.random((K, N), dtype=np.float32)
    #c = np.zeros(shape=(M, N), dtype=np.float32)

    #getArg, getDims = loadAdapter()

    cfg = getDims(M, N, K).contents
    grid = (cfg.gridX, cfg.gridY, cfg.gridZ)
    block = (cfg.blockX, cfg.blockY, cfg.blockZ)

    smem = cfg.smem_size

    #libffCtx.kv.put('a', a)
    aBuf = kaas.bufferSpec('a', a.nbytes, const=False, ephemeral=False)

    #libffCtx.kv.put('b', b)
    bBuf = kaas.bufferSpec('b', b.nbytes, const=False, ephemeral=False)

    #libffCtx.kv.put('c', c, const=False, ephemeral=False)
    cBuf = kaas.bufferSpec('c', c.nbytes, ephemeral=True)
    literals = [kaas.literalSpec('f', alpha), kaas.literalSpec('f', beta),
                kaas.literalSpec('f', M), kaas.literalSpec('f', N), kaas.literalSpec('f', K), kaas.literalSpec('f', lda), kaas.literalSpec('f', ldb), kaas.literalSpec('f', ldc)]
    firstKern = kaas.kernelSpec(kaas.builtins["cutlass"], "sgemm0", grid, block, sharedSize=smem, arguments=[(aBuf, 'i'), (bBuf, 'i'), (cBuf, 'o')], literals=literals)

    #dBuf = kaas.bufferSpec('d', d.nbytes)

    dBuf = kaas.bufferSpec('d', d.nbytes, const=True, ephemeral=False)
    #kv.put('d', d)
    #kv.put('e', e)
    eBuf = kaas.bufferSpec('e', e.nbytes, const=False, ephemeral=False)

    cfg = getDims(M, 1, N).contents
    grid = (cfg.gridX, cfg.gridY, cfg.gridZ)
    block = (cfg.blockX, cfg.blockY, cfg.blockZ)

    smem = cfg.smem_size

    literals = [kaas.literalSpec('f', alpha), kaas.literalSpec('f', beta), kaas.literalSpec('f', M), kaas.literalSpec('f', 1), kaas.literalSpec('f', N), kaas.literalSpec('f', M), kaas.literalSpec('f', N), kaas.literalSpec('f', M)]
    secondKern = kaas.kernelSpec(kaas.builtins["cutlass"], "sgemm0", grid, block, sharedSize=smem, arguments=[(cBuf, 'i'), (dBuf, 'i'), (eBuf, 'o')], literals=literals)


    req = kaas.kaasReq([firstKern, secondKern])
    #kaasHandle = kaas.kaasFF.getHandle("direct", libffCtx)
    #kaasHandle.Invoke(req.toDict())
    return req
    #c = np.frombuffer(libffCtx.kv.get('c'), dtype=np.float32)
    #print(c)


