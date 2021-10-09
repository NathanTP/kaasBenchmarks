import pathlib
import math
import sys
import subprocess as sp
from pprint import pprint
# import GPUtil

import time
import pickle

import libff as ff
import libff.kv
import libff.invoke

import libff.kaas as kaas
import libff.kaas.kaasFF

import numpy as np

redisPwd = "Cd+OBWBEAXV0o2fg5yDrMjD9JUkW7J6MATWuGlRtkQXk/CBvf2HYEjKDYw4FC+eWPeVR8cQKWr7IztZy"
testPath = pathlib.Path(__file__).resolve().parent


def getCtx(remote=False):
    if remote:
        objStore = ff.kv.Redis(pwd=redisPwd, serialize=True)
    else:
        objStore = ff.kv.Local(copyObjs=False, serialize=False)

    return libff.invoke.RemoteCtx(None, objStore)



def createReq(mode='direct'):
    N = 512
    #libffCtx = getCtx(remote=(mode == 'process'))
    #kaasHandle = kaas.kaasFF.getHandle(mode, libffCtx)

    rng = np.random.default_rng(40)
    A = rng.random((N, N), dtype=np.float32)
    fill_arr = np.sum(np.abs(A), axis=1)
    np.fill_diagonal(A, fill_arr)
    b = rng.random((N, 1), dtype=np.float64)

    #libffCtx.kv.put("A", A)
    #libffCtx.kv.put("b", b)

    ABuf = kaas.bufferSpec('A', A.nbytes, key="A")
    bBuf = kaas.bufferSpec('b', b.nbytes, key="b")
    xnewBuf = kaas.bufferSpec('xnew', N*8)
    xBuf = kaas.bufferSpec('x', N*8, ephemeral=True)
    dBuf = kaas.bufferSpec('d', 8)

    arguments1 = [(ABuf, 'i'), (bBuf, 'i'), (xBuf, 'o'), (xnewBuf, 'o'), (dBuf, 'o')]
    arguments2 = [(ABuf, 'i'), (bBuf, 'i'), (xnewBuf, 'o'), (xBuf, 'o'), (dBuf, 'o')]

    kern1 = kaas.kernelSpec(testPath  / 'jacobi.ptx',
                           'JacobiMethod',
                           (256, 1, 1), (66, 1, 1), 8*N,
                           literals=[kaas.literalSpec('i', N)],
                           arguments=arguments1)

    kern2 = kaas.kernelSpec(testPath / 'jacobi.ptx',
                           'JacobiMethod',
                           (256, 1, 1), (66, 1, 1), 8*N,
                           literals=[kaas.literalSpec('i', N)],
                           arguments=arguments2)

    req = kaas.kaasReq([kern1, kern2], nIter=3000)

    # This is just for the test, a real system would use libff to invoke the
    # kaas server
    #kaasHandle.Invoke(req)

    #xnew = libffCtx.kv.get('xnew')
    #xnewArray = np.frombuffer(xnew, dtype=np.float64)
    #print(xnewArray)

    #d = libffCtx.kv.get('d')
    #dArray = np.frombuffer(d, dtype=np.float64)
    #print(dArray)

    #libffCtx.kv.delete("A")
    #libffCtx.kv.delete("b")
    #libffCtx.kv.delete("xnew")
    #libffCtx.kv.delete("d")
    #kaasHandle.Close()

    #return (xnewArray, d)
    return req
