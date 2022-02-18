import pathlib
import pickle
import kaas

testPath = pathlib.Path(__file__).resolve().parent


# Adds the given array to the kv with name node_num.
def addToKV(node_num, nBytes, const=True, ephemeral=False):
    buff = kaas.bufferSpec(str(node_num), nBytes, const=const, ephemeral=ephemeral)
    return buff


def loadParams(param_address):
    params = pickle.load(open(param_address, 'rb'))
    return params


def makeKern(name_func, path, shapes, arguments):
    return kaas.kernelSpec(path, name_func, shapes[0], shapes[1], arguments=arguments)


def createReq(params, cubinPath, mode='direct'):
    nodes = []
    kerns = []
    path = cubinPath
    nodes.append(addToKV(0, 224*224*4, const=False, ephemeral=False))
    # 1. p0
    nodes.append(addToKV(1, params['p0'].nbytes))

    # 2. p1
    nodes.append(addToKV(2, params['p1'].nbytes))

    # 3. fused_nn_conv2d_add_nn_relu_13
    # kernel 0
    output_size = 12845056
    nodes.append(kaas.bufferSpec('3', output_size, const=True, ephemeral=True))
    arguments = [(nodes[0], 'i'), (nodes[1], 'i'), (nodes[2], 'i'), (nodes[3], 'o')]
    shapes = [(14, 112, 1), (16, 1, 4)]
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_13_kernel0', path, shapes, arguments))

    # 4. p2
    nodes.append(addToKV(4, params['p2'].nbytes))

    # 5. p3
    nodes.append(addToKV(5, params['p3'].nbytes))

    # 6. fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_4
    imm = []
    # kernel 0
    imm.append(kaas.bufferSpec('a0', 28901376, const=True, ephemeral=True))
    arguments = [(nodes[3], 'i'), (imm[0], 'o')]
    shapes = [(1568, 1, 1), (128, 1, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_4_kernel0', path, shapes, arguments))

    # kernel 1
    imm.append(kaas.bufferSpec('a1', 28901376, const=True, ephemeral=True))
    arguments = [(nodes[4], 'i'), (imm[0], 'i'), (imm[1], 'o')]
    shapes = [(49, 1, 36), (16, 4, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_4_kernel1', path, shapes, arguments))
    # kernel 2
    output_size = 12845056
    nodes.append(kaas.bufferSpec('6', output_size, const=True, ephemeral=True))
    arguments = [(nodes[5], 'i'), (imm[1], 'i'), (nodes[6], 'o')]
    shapes = [(1568, 1, 1), (128, 1, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_4_kernel2', path, shapes, arguments))

    # 7. p4
    nodes.append(addToKV(7, params['p4'].nbytes))

    # 8. p5
    nodes.append(addToKV(8, params['p5'].nbytes))

    # 9. fused_nn_conv2d_add_nn_relu_12
    # kernel 0
    output_size = 6422528
    nodes.append(kaas.bufferSpec('9', output_size, const=True, ephemeral=True))
    arguments = [(nodes[6], 'i'), (nodes[7], 'i'), (nodes[8], 'i'), (nodes[9], 'o')]
    shapes = [(8, 56, 1), (28, 1, 8)]
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_12_kernel0', path, shapes, arguments))

    # 10. p6
    nodes.append(addToKV(10, params['p6'].nbytes))

    # 11. p7
    nodes.append(addToKV(11, params['p7'].nbytes))

    # 12. fused_nn_conv2d_add_4
    # kernel 0
    output_size = 1806336
    nodes.append(kaas.bufferSpec('12', output_size, const=True, ephemeral=True))
    arguments = [(nodes[9], 'i'), (nodes[10], 'i'), (nodes[11], 'i'), (nodes[12], 'o')]
    shapes = [(14, 28, 1), (8, 8, 3)]
    kerns.append(makeKern('fused_nn_conv2d_add_4_kernel0', path, shapes, arguments))

    # 13. fused_reshape_transpose_reshape
    # kernel 0
    output_size = 1806336
    nodes.append(kaas.bufferSpec('13', output_size, const=False))
    arguments = [(nodes[12], 'i'), (nodes[13], 'o')]
    shapes = [(256, 1, 1), (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    req = kaas.kaasReq(kerns)
    return req
