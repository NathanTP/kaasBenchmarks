import pathlib
import math
from pprint import pprint
import sys
import subprocess as sp
import pickle
import libff as ff
import libff.kv
import libff.invoke
from tvm.contrib.download import download_testdata
from PIL import Image

# import kaasServer as kaas
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


''' Adds the given array to the kv with name node_num. '''
def addToKV(kv, node_num, arr, const=True, ephemeral=False):
    kv.put(str(node_num), arr)
    nByte = arr.nbytes
    buff = kaas.bufferSpec(str(node_num), nByte, const=const, ephemeral=ephemeral)
    return buff 

def loadParams():
    params = pickle.load(open("superRes_params.pkl", 'rb'))
    return params

def makeKern(name_func, path, shapes, arguments):
    return kaas.kernelSpec(path, name_func, shapes[0], shapes[1], arguments=arguments)


def createReq(mode='direct'):    
    libffCtx = getCtx(remote=(mode == 'process'))
    params = loadParams()
    nodes = []
    kerns = []
    path = pathlib.Path(__file__).resolve().parent / 'superRes.cubin'
    inp = np.zeros((1, 1, 224, 224))
    nodes.append(addToKV(libffCtx.kv, 0, inp, const=False, ephemeral=False))

    # 1. p0
    nodes.append(addToKV(libffCtx.kv, 1, params['p0']))


    # 2. p1
    nodes.append(addToKV(libffCtx.kv, 2, params['p1']))


    # 3. fused_nn_conv2d_add_nn_relu_13
    #kernel 0
    output_size = 12845056
    nodes.append(kaas.bufferSpec('3', output_size, const=True, ephemeral=True))
    arguments = [(nodes[0], 'i'), (nodes[1], 'i'), (nodes[2], 'i'), (nodes[3], 'o')]
    shapes = [(14, 112, 1), (16, 1, 4)]
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_13_kernel0', path, shapes, arguments))

    
    # 4. p2
    nodes.append(addToKV(libffCtx.kv, 4, params['p2']))


    # 5. p3
    nodes.append(addToKV(libffCtx.kv, 5, params['p3']))

    # 6. fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_4
    imm = []
    #kernel 0
    #arr = ?
    imm.append(kaas.bufferSpec('a0', 28901376, const=True, ephemeral=True))
    arguments = [(nodes[3], 'i'), (imm[0], 'o')]
    shapes = [(1568, 1, 1), (128, 1, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_4_kernel0', path, shapes, arguments))
    
    #kernel 1
    #arr = ?
    imm.append(kaas.bufferSpec('a1', 28901376, const=True, ephemeral=True))
    arguments = [(nodes[4], 'i'), (imm[0], 'i'), (imm[1], 'o')]
    shapes = [(49, 1, 36), (16, 4, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_4_kernel1', path, shapes, arguments))
    #kernel 2
    output_size = 12845056
    nodes.append(kaas.bufferSpec('6', output_size, const=True, ephemeral=True))
    arguments = [(nodes[5], 'i'), (imm[1], 'i'), (nodes[6], 'o')]
    shapes = [(1568, 1, 1), (128, 1, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_4_kernel2', path, shapes, arguments))
    

    
    # 7. p4
    nodes.append(addToKV(libffCtx.kv, 7, params['p4']))


    # 8. p5
    nodes.append(addToKV(libffCtx.kv, 8, params['p5']))


    # 9. fused_nn_conv2d_add_nn_relu_12
    #kernel 0
    output_size = 6422528
    nodes.append(kaas.bufferSpec('9', output_size, const=True, ephemeral=True))
    arguments = [(nodes[6], 'i'), (nodes[7], 'i'), (nodes[8], 'i'), (nodes[9], 'o')]
    shapes = [(8, 56, 1), (28, 1, 8)] 
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_12_kernel0', path, shapes, arguments))

    
    # 10. p6
    nodes.append(addToKV(libffCtx.kv, 10, params['p6']))


    # 11. p7
    nodes.append(addToKV(libffCtx.kv, 11, params['p7']))


    # 12. fused_nn_conv2d_add_4
    #kernel 0
    output_size = 1806336
    nodes.append(kaas.bufferSpec('12', output_size, const=True, ephemeral=True))
    arguments = [(nodes[9], 'i'), (nodes[10], 'i'), (nodes[11], 'i'), (nodes[12], 'o')]
    shapes = [(14, 28, 1), (8, 8, 3)]
    kerns.append(makeKern('fused_nn_conv2d_add_4_kernel0', path, shapes, arguments))

    # 13. fused_reshape_transpose_reshape
    #kernel 0
    output_size = 1806336
    nodes.append(kaas.bufferSpec('13', output_size, const=False))
    arguments = [(nodes[12], 'i'), (nodes[13], 'o')]
    shapes = [(256, 1, 1), (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))


    req = kaas.kaasReq(kerns)

    return req 





  

