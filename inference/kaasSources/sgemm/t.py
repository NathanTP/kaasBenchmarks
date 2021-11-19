import libff as ff
import libff.kv
import libff.invoke
import libff.kaas as kaas
import libff.kaas.kaasFF

import numpy as np
import yaml

import pycuda.driver as cuda
import pycuda.autoinit  # NOQA

matSize = 256
# matSize = 128
matNByte = (matSize*matSize*4)


def cudaTest():
    mod = cuda.module_from_file("kerns/sgemm.cubin")
    # kern = mod.get_function("sgemm")
    kern = mod.get_function("matmul")
    kern.prepare(["P", "P", "P"])

    # tileN = 16
    # tile_tb_height = 8
    # tileM = (tileN * tile_tb_height)
    # gridDim = (matSize // tileM, matSize // tileN)
    # blockDim = (tileN, tile_tb_height, 1)
    tile = 8
    gridDim = (matSize // tile, matSize // tile, 1)
    blockDim = (tile, tile, 1)

    a = np.full((matSize, matSize), 10, dtype=np.float32)
    b = np.identity(matSize, dtype=np.float32)
    c = np.empty_like(a)

    da = cuda.mem_alloc(matNByte)
    db = cuda.mem_alloc(matNByte)
    dc = cuda.mem_alloc(matNByte)

    cuda.memcpy_htod(da, a)
    cuda.memcpy_htod(db, b)
    cuda.memset_d8(dc, 0, matNByte)

    # kern.prepared_call(gridDim, blockDim, da, db, dc, shared_size=sharedSize)
    kern.prepared_call(gridDim, blockDim, da, db, dc)

    cuda.memcpy_dtoh(c, dc)
    print(c[0])


def kaasTest():
    objStore = ff.kv.Local(copyObjs=False, serialize=False)
    ctx = ff.invoke.RemoteCtx(None, objStore)
    kaasHandle = kaas.kaasFF.getHandle('direct', ctx)

    with open("sgemm_model.yaml", 'r') as f:
        rDict = yaml.safe_load(f)

    req = kaas.kaasReqDense.fromDict(rDict)

    shape = (128, 128)
    inputA = np.full(shape, 10, dtype=np.float32)

    const0 = np.zeros(shape, dtype=np.float32)
    np.fill_diagonal(const0, 1)

    const1 = np.zeros(shape, dtype=np.float32)
    np.fill_diagonal(const1, 2)

    const2 = np.zeros(shape, dtype=np.float32)
    np.fill_diagonal(const2, 3)

    ctx.kv.put("inputA", inputA)
    ctx.kv.put("inputB", const0)
    ctx.kv.put("intermediate0B", const1)
    ctx.kv.put("outputB", const2)

    kaasHandle.Invoke(req)

    cRaw = ctx.kv.get('outputC')
    cArr = np.frombuffer(cRaw, dtype=np.float32)
    testRes = cArr.reshape(128, 128)

    expect = np.matmul(inputA, const0)
    expect = np.matmul(expect, const1)
    expect = np.matmul(expect, const2)

    if np.allclose(expect, testRes, rtol=0.05):
        print("PASS")
    else:
        print("FAIL")
        dist = np.linalg.norm(expect - testRes)
        print("Returned matrix doesn't look right: l2=", dist)


cudaTest()
