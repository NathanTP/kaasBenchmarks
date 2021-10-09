from libff import kaas
import numpy as np
import ctypes as ct
import libff as ff
import libff.kv
import libff.invoke
import libff.kaas.kaasFF
import libff.kaas as kaas


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
    aBuf = kaas.bufferSpec('a', a.nbytes, ephemeral=False)

    #libffCtx.kv.put('b', b)
    bBuf = kaas.bufferSpec('b', b.nbytes, ephemeral=False)

    #libffCtx.kv.put('c', c)
    cBuf = kaas.bufferSpec('c', c.nbytes, ephemeral=True)
    literals = [kaas.literalSpec('f', alpha), kaas.literalSpec('f', beta),
                kaas.literalSpec('f', M), kaas.literalSpec('f', N), kaas.literalSpec('f', K), kaas.literalSpec('f', lda), kaas.literalSpec('f', ldb), kaas.literalSpec('f', ldc)]
    firstKern = kaas.kernelSpec(kaas.builtins["complexCutlass"], "complexGemm0", grid, block, sharedSize=smem, arguments=[(aBuf, 'i'), (bBuf, 'i'), (cBuf, 'o')], literals=literals)

    dBuf = kaas.bufferSpec('d', d.nbytes)

    dBuf = kaas.bufferSpec('d', d.nbytes, ephemeral=False)
    #libffCtx.kv.put('d', d)
    #libffCtx.kv.put('e', e)
    eBuf = kaas.bufferSpec('e', e.nbytes, ephemeral=False)

    cfg = getDims(M, 1, N).contents
    grid = (cfg.gridX, cfg.gridY, cfg.gridZ)
    block = (cfg.blockX, cfg.blockY, cfg.blockZ)

    smem = cfg.smem_size

    literals = [kaas.literalSpec('f', alpha), kaas.literalSpec('f', beta), kaas.literalSpec('f', M), kaas.literalSpec('f', 1), kaas.literalSpec('f', N), kaas.literalSpec('f', M), kaas.literalSpec('f', N), kaas.literalSpec('f', M)]
    secondKern = kaas.kernelSpec(kaas.builtins["complexCutlass"], "complexGemm0", grid, block, sharedSize=smem, arguments=[(cBuf, 'i'), (dBuf, 'i'), (eBuf, 'o')], literals=literals)

    req = kaas.kaasReq([firstKern, secondKern])
    return req
    #req = kaas.kaasReqDense([firstKern, secondKern])
    #kaasHandle = kaas.kaasFF.getHandle("direct", libffCtx)
    #kaasHandle.Invoke(req)
    #return req
    #e = np.frombuffer(libffCtx.kv.get('e'), dtype=np.csingle)
    #e = np.reshape(e, (M, 1), order='F')
    #print(e)
    #print(c)

