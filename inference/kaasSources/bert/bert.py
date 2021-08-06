import pathlib
import pickle
import numpy as np
# import kaasServer as kaas
import libff.kaas as kaas

redisPwd = "Cd+OBWBEAXV0o2fg5yDrMjD9JUkW7J6MATWuGlRtkQXk/CBvf2HYEjKDYw4FC+eWPeVR8cQKWr7IztZy"
testPath = pathlib.Path(__file__).resolve().parent


# Adds the given array to the kv with name node_num.
def addToKV(node_num, arr, const=True, ephemeral=False):
    #kv.put(str(node_num), arr)
    nByte = arr.nbytes
    buff = kaas.bufferSpec(str(node_num), nByte, const=const, ephemeral=ephemeral)
    return buff


def loadParams():
    path = pathlib.Path.cwd() / "bert" / "bert_params.pkl"
    params = pickle.load(open(path, 'rb'))
    return {'p' + str(i): params[i] for i in range(len(params))}


def makeKern(name_func, path, shapes, arguments):
    return kaas.kernelSpec(path, name_func, shapes[0], shapes[1], arguments=arguments)

'''
kv = None
def runReq():
    libffCtx = getCtx(remote=False)
    global kv
    kv = libffCtx.kv
    from infbench import bert
    import numpy as np
    loader = bert.bertLoader(pathlib.Path.cwd() / "bertData")

    inputs = loader.get(0)


    constants = bert.bertModel.getConstants(pathlib.Path.cwd())

    pre_input = constants + [inputs[0]]

    pre_output = bert.bertModel.pre(pre_input)


    graph_inputs = []
    graph_inputs.append(np.frombuffer(pre_output[0], dtype=np.int64))
    graph_inputs.append(np.frombuffer(pre_output[1], dtype=np.int64))
    graph_inputs.append(np.frombuffer(pre_output[2], dtype=np.int64))

    req = createReq(graph_inputs)

    mode = "direct"
    kaasHandle = kaas.kaasFF.getHandle(mode, libffCtx)
    kaasHandle.Invoke(req.toDict())

    c = np.frombuffer(libffCtx.kv.get('1123'), dtype=np.float32)
    print(c)
'''

def createReq(params, path, mode='direct'):
    nodes = dict()
    kerns = []
    nodes[0] = addToKV(0, np.zeros((1, 384)), const=False)
    # storage = dict()
    # storage['0'] = nodes[0]

    # 1. input_mask
    nodes[1] = addToKV(1, np.zeros((1, 384)), const=False)

    # 2. segment_ids
    nodes[2] = addToKV(2, np.zeros((1, 384)), const=False)

    # 3. p0
    nodes[3] = addToKV(3, params['p0'])

    # 4. p1
    nodes[4] = addToKV(4, params['p1'])

    # 5. p2
    nodes[5] = addToKV(5, params['p2'])

    # 6. fused_less_add_where_take_add_less_add_where_take_add
    # kernel 0
    output_size = 1572864
    nodes[6] = kaas.bufferSpec('6', output_size, const=False, ephemeral=True)
    arguments = [(nodes[6], 'o'), (nodes[3], 'i'), (nodes[0], 'i'), (nodes[4], 'i'), (nodes[5], 'i'), (nodes[2], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_less_add_where_take_add_less_add_where_take_add_kernel0', path, shapes, arguments))

    # 7. fused_mean
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a0', output_size, const=False, ephemeral=True))
    arguments = [(nodes[6], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[7] = kaas.bufferSpec('7', output_size, const=False, ephemeral=True)
    arguments = [(nodes[7], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 8. fused_subtract48
    # kernel 0
    output_size = 1572864
    nodes[8] = kaas.bufferSpec('8', output_size, const=False, ephemeral=True)
    arguments = [(nodes[8], 'o'), (nodes[6], 'i'), (nodes[7], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 9. fused_power_mean48
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a1', output_size, const=False, ephemeral=True))
    arguments = [(nodes[8], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[9] = kaas.bufferSpec('9', output_size, const=False, ephemeral=True)
    arguments = [(nodes[9], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 10. p3
    nodes[10] = addToKV(10, params['p3'])

    # 11. p4
    nodes[11] = addToKV(11, params['p4'])

    # 12. fused_add_sqrt_divide_multiply_add47
    # kernel 0
    output_size = 1572864
    nodes[12] = kaas.bufferSpec('12', output_size, const=False, ephemeral=True)
    arguments = [(nodes[12], 'o'), (nodes[9], 'i'), (nodes[8], 'i'), (nodes[10], 'i'), (nodes[11], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 13. reshape_nop
    nodes[13] = nodes[12]

    # 14. p5
    nodes[14] = addToKV(14, params['p5'])

    # 15. fused_nn_batch_matmul_347
    # kernel 0
    output_size = 1572864
    nodes[15] = kaas.bufferSpec('15', output_size, const=False, ephemeral=True)
    arguments = [(nodes[12], 'i'), (nodes[14], 'i'), (nodes[15], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 16. p6
    nodes[16] = addToKV(16, params['p6'])

    # 17. fused_reshape_add_reshape_transpose_reshape23
    # kernel 0
    output_size = 1572864
    nodes[17] = kaas.bufferSpec('17', output_size, const=False, ephemeral=True)
    arguments = [(nodes[17], 'o'), (nodes[15], 'i'), (nodes[16], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 18. p7
    nodes[18] = addToKV(18, params['p7'])

    # 19. fused_nn_batch_matmul_348
    # kernel 0
    output_size = 1572864
    nodes[19] = kaas.bufferSpec('19', output_size, const=False, ephemeral=True)
    arguments = [(nodes[13], 'i'), (nodes[18], 'i'), (nodes[19], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 20. p8
    nodes[20] = addToKV(20, params['p8'])

    # 21. fused_reshape_add_reshape_transpose_reshape_transpose
    # kernel 0
    output_size = 1572864
    nodes[21] = kaas.bufferSpec('21', output_size, const=False, ephemeral=True)
    arguments = [(nodes[21], 'o'), (nodes[19], 'i'), (nodes[20], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 22. fused_nn_batch_matmul_523
    # kernel 0
    output_size = 9437184
    nodes[22] = kaas.bufferSpec('22', output_size, const=False, ephemeral=True)
    arguments = [(nodes[17], 'i'), (nodes[21], 'i'), (nodes[22], 'o'), ]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 23. fused_expand_dims_expand_dims_cast_subtract_multiply
    # kernel 0
    output_size = 1536
    nodes[23] = kaas.bufferSpec('23', output_size, const=False, ephemeral=True)
    arguments = [(nodes[23], 'o'), (nodes[1], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_expand_dims_expand_dims_cast_subtract_multiply_kernel0', path, shapes, arguments))

    # 24. fused_reshape_divide_add23
    # kernel 0
    output_size = 9437184
    nodes[24] = kaas.bufferSpec('24', output_size, const=False, ephemeral=True)
    arguments = [(nodes[24], 'o'), (nodes[22], 'i'), (nodes[23], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 25. fused_max
    # kernel 0
    output_size = 24576
    nodes[25] = kaas.bufferSpec('25', output_size, const=False, ephemeral=True)
    arguments = [(nodes[24], 'i'), (nodes[25], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 26. fused_subtract_exp23
    # kernel 0
    output_size = 9437184
    nodes[26] = kaas.bufferSpec('26', output_size, const=False, ephemeral=True)
    arguments = [(nodes[26], 'o'), (nodes[24], 'i'), (nodes[25], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 27. fused_sum
    # kernel 0
    output_size = 24576
    nodes[27] = kaas.bufferSpec('27', output_size, const=False, ephemeral=True)
    arguments = [(nodes[26], 'i'), (nodes[27], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 28. fused_divide_reshape23
    # kernel 0
    output_size = 9437184
    nodes[28] = kaas.bufferSpec('28', output_size, const=False, ephemeral=True)
    arguments = [(nodes[28], 'o'), (nodes[26], 'i'), (nodes[27], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 29. p9
    nodes[29] = addToKV(29, params['p9'])

    # 30. fused_nn_batch_matmul_349
    # kernel 0
    output_size = 1572864
    nodes[30] = kaas.bufferSpec('30', output_size, const=False, ephemeral=True)
    arguments = [(nodes[13], 'i'), (nodes[29], 'i'), (nodes[30], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 31. p10
    nodes[31] = addToKV(31, params['p10'])

    # 32. fused_reshape_add_reshape_transpose_reshape_transpose_1
    # kernel 0
    output_size = 1572864
    nodes[32] = kaas.bufferSpec('32', output_size, const=False, ephemeral=True)
    arguments = [(nodes[32], 'o'), (nodes[30], 'i'), (nodes[31], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 33. fused_nn_batch_matmul_423
    # kernel 0
    output_size = 1572864
    nodes[33] = kaas.bufferSpec('33', output_size, const=False, ephemeral=True)
    arguments = [(nodes[28], 'i'), (nodes[32], 'i'), (nodes[33], 'o'), ]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 34. fused_reshape_transpose_reshape23
    # kernel 0
    output_size = 1572864
    nodes[34] = kaas.bufferSpec('34', output_size, const=False, ephemeral=True)
    arguments = [(nodes[34], 'o'), (nodes[33], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 35. p11
    nodes[35] = addToKV(35, params['p11'])

    # 36. fused_nn_batch_matmul_346
    # kernel 0
    output_size = 1572864
    nodes[36] = kaas.bufferSpec('36', output_size, const=False, ephemeral=True)
    arguments = [(nodes[34], 'i'), (nodes[35], 'i'), (nodes[36], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 37. p12
    nodes[37] = addToKV(37, params['p12'])

    # 38. fused_reshape_add_add47
    # kernel 0
    output_size = 1572864
    nodes[38] = kaas.bufferSpec('38', output_size, const=False, ephemeral=True)
    arguments = [(nodes[38], 'o'), (nodes[36], 'i'), (nodes[37], 'i'), (nodes[12], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 39. fused_mean1
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a2', output_size, const=False, ephemeral=True))
    arguments = [(nodes[38], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[39] = kaas.bufferSpec('39', output_size, const=False, ephemeral=True)
    arguments = [(nodes[39], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 40. fused_subtract47
    # kernel 0
    output_size = 1572864
    nodes[40] = kaas.bufferSpec('40', output_size, const=False, ephemeral=True)
    arguments = [(nodes[40], 'o'), (nodes[38], 'i'), (nodes[39], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 41. fused_power_mean47
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a3', output_size, const=False, ephemeral=True))
    arguments = [(nodes[40], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[41] = kaas.bufferSpec('41', output_size, const=False, ephemeral=True)
    arguments = [(nodes[41], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 42. p13
    nodes[42] = addToKV(42, params['p13'])

    # 43. p14
    nodes[43] = addToKV(43, params['p14'])

    # 44. fused_add_sqrt_divide_multiply_add46
    # kernel 0
    output_size = 1572864
    nodes[44] = kaas.bufferSpec('44', output_size, const=False, ephemeral=True)
    arguments = [(nodes[44], 'o'), (nodes[41], 'i'), (nodes[40], 'i'), (nodes[42], 'i'), (nodes[43], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 45. reshape_nop
    nodes[45] = nodes[44]

    # 46. p15
    nodes[46] = addToKV(46, params['p15'])

    # 47. fused_nn_batch_matmul_223
    # kernel 0
    output_size = 6291456
    nodes[47] = kaas.bufferSpec('47', output_size, const=False, ephemeral=True)
    arguments = [(nodes[45], 'i'), (nodes[46], 'i'), (nodes[47], 'o'), ]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 48. p16
    nodes[48] = addToKV(48, params['p16'])

    # 49. fused_reshape_add_multiply_divide_erf_add_multiply_reshape23
    # kernel 0
    output_size = 6291456
    nodes[49] = kaas.bufferSpec('49', output_size, const=False, ephemeral=True)
    arguments = [(nodes[49], 'o'), (nodes[47], 'i'), (nodes[48], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 50. p17
    nodes[50] = addToKV(50, params['p17'])

    # 51. fused_nn_batch_matmul_123
    # kernel 0
    output_size = 1572864
    nodes[51] = kaas.bufferSpec('51', output_size, const=False, ephemeral=True)
    arguments = [(nodes[49], 'i'), (nodes[50], 'i'), (nodes[51], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 52. p18
    nodes[52] = addToKV(52, params['p18'])

    # 53. fused_reshape_add_add46
    # kernel 0
    output_size = 1572864
    nodes[53] = kaas.bufferSpec('53', output_size, const=False, ephemeral=True)
    arguments = [(nodes[53], 'o'), (nodes[51], 'i'), (nodes[52], 'i'), (nodes[44], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 54. fused_mean2
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a4', output_size, const=False, ephemeral=True))
    arguments = [(nodes[53], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[54] = kaas.bufferSpec('54', output_size, const=False, ephemeral=True)
    arguments = [(nodes[54], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 55. fused_subtract46
    # kernel 0
    output_size = 1572864
    nodes[55] = kaas.bufferSpec('55', output_size, const=False, ephemeral=True)
    arguments = [(nodes[55], 'o'), (nodes[53], 'i'), (nodes[54], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 56. fused_power_mean46
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a5', output_size, const=False, ephemeral=True))
    arguments = [(nodes[55], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[56] = kaas.bufferSpec('56', output_size, const=False, ephemeral=True)
    arguments = [(nodes[56], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 57. p19
    nodes[57] = addToKV(57, params['p19'])

    # 58. p20
    nodes[58] = addToKV(58, params['p20'])

    # 59. fused_add_sqrt_divide_multiply_add45
    # kernel 0
    output_size = 1572864
    nodes[59] = kaas.bufferSpec('59', output_size, const=False, ephemeral=True)
    arguments = [(nodes[59], 'o'), (nodes[56], 'i'), (nodes[55], 'i'), (nodes[57], 'i'), (nodes[58], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 60. reshape_nop
    nodes[60] = nodes[59]

    # 61. p21
    nodes[61] = addToKV(61, params['p21'])

    # 62. fused_nn_batch_matmul_345
    # kernel 0
    output_size = 1572864
    nodes[62] = kaas.bufferSpec('62', output_size, const=False, ephemeral=True)
    arguments = [(nodes[60], 'i'), (nodes[61], 'i'), (nodes[62], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 63. p22
    nodes[63] = addToKV(63, params['p22'])

    # 64. fused_reshape_add_reshape_transpose_reshape22
    # kernel 0
    output_size = 1572864
    nodes[64] = kaas.bufferSpec('64', output_size, const=False, ephemeral=True)
    arguments = [(nodes[64], 'o'), (nodes[62], 'i'), (nodes[63], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 65. p23
    nodes[65] = addToKV(65, params['p23'])

    # 66. fused_nn_batch_matmul_350
    # kernel 0
    output_size = 1572864
    nodes[66] = kaas.bufferSpec('66', output_size, const=False, ephemeral=True)
    arguments = [(nodes[60], 'i'), (nodes[65], 'i'), (nodes[66], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 67. p24
    nodes[67] = addToKV(67, params['p24'])

    # 68. fused_reshape_add_reshape_transpose_reshape_transpose1
    # kernel 0
    output_size = 1572864
    nodes[68] = kaas.bufferSpec('68', output_size, const=False, ephemeral=True)
    arguments = [(nodes[68], 'o'), (nodes[66], 'i'), (nodes[67], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 69. fused_nn_batch_matmul_522
    # kernel 0
    output_size = 9437184
    nodes[69] = kaas.bufferSpec('69', output_size, const=False, ephemeral=True)
    arguments = [(nodes[64], 'i'), (nodes[68], 'i'), (nodes[69], 'o'), ]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 70. fused_reshape_divide_add22
    # kernel 0
    output_size = 9437184
    nodes[70] = kaas.bufferSpec('70', output_size, const=False, ephemeral=True)
    arguments = [(nodes[70], 'o'), (nodes[69], 'i'), (nodes[23], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 71. fused_max1
    # kernel 0
    output_size = 24576
    nodes[71] = kaas.bufferSpec('71', output_size, const=False, ephemeral=True)
    arguments = [(nodes[70], 'i'), (nodes[71], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 72. fused_subtract_exp22
    # kernel 0
    output_size = 9437184
    nodes[72] = kaas.bufferSpec('72', output_size, const=False, ephemeral=True)
    arguments = [(nodes[72], 'o'), (nodes[70], 'i'), (nodes[71], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 73. fused_sum1
    # kernel 0
    output_size = 24576
    nodes[73] = kaas.bufferSpec('73', output_size, const=False, ephemeral=True)
    arguments = [(nodes[72], 'i'), (nodes[73], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 74. fused_divide_reshape22
    # kernel 0
    output_size = 9437184
    nodes[74] = kaas.bufferSpec('74', output_size, const=False, ephemeral=True)
    arguments = [(nodes[74], 'o'), (nodes[72], 'i'), (nodes[73], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 75. p25
    nodes[75] = addToKV(75, params['p25'])

    # 76. fused_nn_batch_matmul_351
    # kernel 0
    output_size = 1572864
    nodes[76] = kaas.bufferSpec('76', output_size, const=False, ephemeral=True)
    arguments = [(nodes[60], 'i'), (nodes[75], 'i'), (nodes[76], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 77. p26
    nodes[77] = addToKV(77, params['p26'])

    # 78. fused_reshape_add_reshape_transpose_reshape_transpose_11
    # kernel 0
    output_size = 1572864
    nodes[78] = kaas.bufferSpec('78', output_size, const=False, ephemeral=True)
    arguments = [(nodes[78], 'o'), (nodes[76], 'i'), (nodes[77], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 79. fused_nn_batch_matmul_422
    # kernel 0
    output_size = 1572864
    nodes[79] = kaas.bufferSpec('79', output_size, const=False, ephemeral=True)
    arguments = [(nodes[74], 'i'), (nodes[78], 'i'), (nodes[79], 'o'), ]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 80. fused_reshape_transpose_reshape22
    # kernel 0
    output_size = 1572864
    nodes[80] = kaas.bufferSpec('80', output_size, const=False, ephemeral=True)
    arguments = [(nodes[80], 'o'), (nodes[79], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 81. p27
    nodes[81] = addToKV(81, params['p27'])

    # 82. fused_nn_batch_matmul_344
    # kernel 0
    output_size = 1572864
    nodes[82] = kaas.bufferSpec('82', output_size, const=False, ephemeral=True)
    arguments = [(nodes[80], 'i'), (nodes[81], 'i'), (nodes[82], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 83. p28
    nodes[83] = addToKV(83, params['p28'])

    # 84. fused_reshape_add_add45
    # kernel 0
    output_size = 1572864
    nodes[84] = kaas.bufferSpec('84', output_size, const=False, ephemeral=True)
    arguments = [(nodes[84], 'o'), (nodes[82], 'i'), (nodes[83], 'i'), (nodes[59], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 85. fused_mean3
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a6', output_size, const=False, ephemeral=True))
    arguments = [(nodes[84], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[85] = kaas.bufferSpec('85', output_size, const=False, ephemeral=True)
    arguments = [(nodes[85], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 86. fused_subtract45
    # kernel 0
    output_size = 1572864
    nodes[86] = kaas.bufferSpec('86', output_size, const=False, ephemeral=True)
    arguments = [(nodes[86], 'o'), (nodes[84], 'i'), (nodes[85], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 87. fused_power_mean45
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a7', output_size, const=False, ephemeral=True))
    arguments = [(nodes[86], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[87] = kaas.bufferSpec('87', output_size, const=False, ephemeral=True)
    arguments = [(nodes[87], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 88. p29
    nodes[88] = addToKV(88, params['p29'])

    # 89. p30
    nodes[89] = addToKV(89, params['p30'])

    # 90. fused_add_sqrt_divide_multiply_add44
    # kernel 0
    output_size = 1572864
    nodes[90] = kaas.bufferSpec('90', output_size, const=False, ephemeral=True)
    arguments = [(nodes[90], 'o'), (nodes[87], 'i'), (nodes[86], 'i'), (nodes[88], 'i'), (nodes[89], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 91. reshape_nop
    nodes[91] = nodes[90]

    # 92. p31
    nodes[92] = addToKV(92, params['p31'])

    # 93. fused_nn_batch_matmul_222
    # kernel 0
    output_size = 6291456
    nodes[93] = kaas.bufferSpec('93', output_size, const=False, ephemeral=True)
    arguments = [(nodes[91], 'i'), (nodes[92], 'i'), (nodes[93], 'o'), ]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 94. p32
    nodes[94] = addToKV(94, params['p32'])

    # 95. fused_reshape_add_multiply_divide_erf_add_multiply_reshape22
    # kernel 0
    output_size = 6291456
    nodes[95] = kaas.bufferSpec('95', output_size, const=False, ephemeral=True)
    arguments = [(nodes[95], 'o'), (nodes[93], 'i'), (nodes[94], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 96. p33
    nodes[96] = addToKV(96, params['p33'])

    # 97. fused_nn_batch_matmul_122
    # kernel 0
    output_size = 1572864
    nodes[97] = kaas.bufferSpec('97', output_size, const=False, ephemeral=True)
    arguments = [(nodes[95], 'i'), (nodes[96], 'i'), (nodes[97], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 98. p34
    nodes[98] = addToKV(98, params['p34'])

    # 99. fused_reshape_add_add44
    # kernel 0
    output_size = 1572864
    nodes[99] = kaas.bufferSpec('99', output_size, const=False, ephemeral=True)
    arguments = [(nodes[99], 'o'), (nodes[97], 'i'), (nodes[98], 'i'), (nodes[90], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 100. fused_mean4
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a8', output_size, const=False, ephemeral=True))
    arguments = [(nodes[99], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[100] = kaas.bufferSpec('100', output_size, const=False, ephemeral=True)
    arguments = [(nodes[100], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 101. fused_subtract44
    # kernel 0
    output_size = 1572864
    nodes[101] = kaas.bufferSpec('101', output_size, const=False, ephemeral=True)
    arguments = [(nodes[101], 'o'), (nodes[99], 'i'), (nodes[100], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 102. fused_power_mean44
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a9', output_size, const=False, ephemeral=True))
    arguments = [(nodes[101], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[102] = kaas.bufferSpec('102', output_size, const=False, ephemeral=True)
    arguments = [(nodes[102], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 103. p35
    nodes[103] = addToKV(103, params['p35'])

    # 104. p36
    nodes[104] = addToKV(104, params['p36'])

    # 105. fused_add_sqrt_divide_multiply_add43
    # kernel 0
    output_size = 1572864
    nodes[105] = kaas.bufferSpec('105', output_size, const=False, ephemeral=True)
    arguments = [(nodes[105], 'o'), (nodes[102], 'i'), (nodes[101], 'i'), (nodes[103], 'i'), (nodes[104], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 106. reshape_nop
    nodes[106] = nodes[105]

    # 107. p37
    nodes[107] = addToKV(107, params['p37'])

    # 108. fused_nn_batch_matmul_343
    # kernel 0
    output_size = 1572864
    nodes[108] = kaas.bufferSpec('108', output_size, const=False, ephemeral=True)
    arguments = [(nodes[106], 'i'), (nodes[107], 'i'), (nodes[108], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 109. p38
    nodes[109] = addToKV(109, params['p38'])

    # 110. fused_reshape_add_reshape_transpose_reshape21
    # kernel 0
    output_size = 1572864
    nodes[110] = kaas.bufferSpec('110', output_size, const=False, ephemeral=True)
    arguments = [(nodes[110], 'o'), (nodes[108], 'i'), (nodes[109], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 111. p39
    nodes[111] = addToKV(111, params['p39'])

    # 112. fused_nn_batch_matmul_352
    # kernel 0
    output_size = 1572864
    nodes[112] = kaas.bufferSpec('112', output_size, const=False, ephemeral=True)
    arguments = [(nodes[106], 'i'), (nodes[111], 'i'), (nodes[112], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 113. p40
    nodes[113] = addToKV(113, params['p40'])

    # 114. fused_reshape_add_reshape_transpose_reshape_transpose2
    # kernel 0
    output_size = 1572864
    nodes[114] = kaas.bufferSpec('114', output_size, const=False, ephemeral=True)
    arguments = [(nodes[114], 'o'), (nodes[112], 'i'), (nodes[113], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 115. fused_nn_batch_matmul_521
    # kernel 0
    output_size = 9437184
    nodes[115] = kaas.bufferSpec('115', output_size, const=False, ephemeral=True)
    arguments = [(nodes[110], 'i'), (nodes[114], 'i'), (nodes[115], 'o'), ]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 116. fused_reshape_divide_add21
    # kernel 0
    output_size = 9437184
    nodes[116] = kaas.bufferSpec('116', output_size, const=False, ephemeral=True)
    arguments = [(nodes[116], 'o'), (nodes[115], 'i'), (nodes[23], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 117. fused_max2
    # kernel 0
    output_size = 24576
    nodes[117] = kaas.bufferSpec('117', output_size, const=False, ephemeral=True)
    arguments = [(nodes[116], 'i'), (nodes[117], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 118. fused_subtract_exp21
    # kernel 0
    output_size = 9437184
    nodes[118] = kaas.bufferSpec('118', output_size, const=False, ephemeral=True)
    arguments = [(nodes[118], 'o'), (nodes[116], 'i'), (nodes[117], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 119. fused_sum2
    # kernel 0
    output_size = 24576
    nodes[119] = kaas.bufferSpec('119', output_size, const=False, ephemeral=True)
    arguments = [(nodes[118], 'i'), (nodes[119], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 120. fused_divide_reshape21
    # kernel 0
    output_size = 9437184
    nodes[120] = kaas.bufferSpec('120', output_size, const=False, ephemeral=True)
    arguments = [(nodes[120], 'o'), (nodes[118], 'i'), (nodes[119], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 121. p41
    nodes[121] = addToKV(121, params['p41'])

    # 122. fused_nn_batch_matmul_353
    # kernel 0
    output_size = 1572864
    nodes[122] = kaas.bufferSpec('122', output_size, const=False, ephemeral=True)
    arguments = [(nodes[106], 'i'), (nodes[121], 'i'), (nodes[122], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 123. p42
    nodes[123] = addToKV(123, params['p42'])

    # 124. fused_reshape_add_reshape_transpose_reshape_transpose_12
    # kernel 0
    output_size = 1572864
    nodes[124] = kaas.bufferSpec('124', output_size, const=False, ephemeral=True)
    arguments = [(nodes[124], 'o'), (nodes[122], 'i'), (nodes[123], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 125. fused_nn_batch_matmul_421
    # kernel 0
    output_size = 1572864
    nodes[125] = kaas.bufferSpec('125', output_size, const=False, ephemeral=True)
    arguments = [(nodes[120], 'i'), (nodes[124], 'i'), (nodes[125], 'o'), ]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 126. fused_reshape_transpose_reshape21
    # kernel 0
    output_size = 1572864
    nodes[126] = kaas.bufferSpec('126', output_size, const=False, ephemeral=True)
    arguments = [(nodes[126], 'o'), (nodes[125], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 127. p43
    nodes[127] = addToKV(127, params['p43'])

    # 128. fused_nn_batch_matmul_342
    # kernel 0
    output_size = 1572864
    nodes[128] = kaas.bufferSpec('128', output_size, const=False, ephemeral=True)
    arguments = [(nodes[126], 'i'), (nodes[127], 'i'), (nodes[128], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 129. p44
    nodes[129] = addToKV(129, params['p44'])

    # 130. fused_reshape_add_add43
    # kernel 0
    output_size = 1572864
    nodes[130] = kaas.bufferSpec('130', output_size, const=False, ephemeral=True)
    arguments = [(nodes[130], 'o'), (nodes[128], 'i'), (nodes[129], 'i'), (nodes[105], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 131. fused_mean5
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a10', output_size, const=False, ephemeral=True))
    arguments = [(nodes[130], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[131] = kaas.bufferSpec('131', output_size, const=False, ephemeral=True)
    arguments = [(nodes[131], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 132. fused_subtract43
    # kernel 0
    output_size = 1572864
    nodes[132] = kaas.bufferSpec('132', output_size, const=False, ephemeral=True)
    arguments = [(nodes[132], 'o'), (nodes[130], 'i'), (nodes[131], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 133. fused_power_mean43
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a11', output_size, const=False, ephemeral=True))
    arguments = [(nodes[132], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[133] = kaas.bufferSpec('133', output_size, const=False, ephemeral=True)
    arguments = [(nodes[133], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 134. p45
    nodes[134] = addToKV(134, params['p45'])

    # 135. p46
    nodes[135] = addToKV(135, params['p46'])

    # 136. fused_add_sqrt_divide_multiply_add42
    # kernel 0
    output_size = 1572864
    nodes[136] = kaas.bufferSpec('136', output_size, const=False, ephemeral=True)
    arguments = [(nodes[136], 'o'), (nodes[133], 'i'), (nodes[132], 'i'), (nodes[134], 'i'), (nodes[135], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 137. reshape_nop
    nodes[137] = nodes[136]

    # 138. p47
    nodes[138] = addToKV(138, params['p47'])

    # 139. fused_nn_batch_matmul_221
    # kernel 0
    output_size = 6291456
    nodes[139] = kaas.bufferSpec('139', output_size, const=False, ephemeral=True)
    arguments = [(nodes[137], 'i'), (nodes[138], 'i'), (nodes[139], 'o'), ]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 140. p48
    nodes[140] = addToKV(140, params['p48'])

    # 141. fused_reshape_add_multiply_divide_erf_add_multiply_reshape21
    # kernel 0
    output_size = 6291456
    nodes[141] = kaas.bufferSpec('141', output_size, const=False, ephemeral=True)
    arguments = [(nodes[141], 'o'), (nodes[139], 'i'), (nodes[140], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 142. p49
    nodes[142] = addToKV(142, params['p49'])

    # 143. fused_nn_batch_matmul_121
    # kernel 0
    output_size = 1572864
    nodes[143] = kaas.bufferSpec('143', output_size, const=False, ephemeral=True)
    arguments = [(nodes[141], 'i'), (nodes[142], 'i'), (nodes[143], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 144. p50
    nodes[144] = addToKV(144, params['p50'])

    # 145. fused_reshape_add_add42
    # kernel 0
    output_size = 1572864
    nodes[145] = kaas.bufferSpec('145', output_size, const=False, ephemeral=True)
    arguments = [(nodes[145], 'o'), (nodes[143], 'i'), (nodes[144], 'i'), (nodes[136], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 146. fused_mean6
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a12', output_size, const=False, ephemeral=True))
    arguments = [(nodes[145], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[146] = kaas.bufferSpec('146', output_size, const=False, ephemeral=True)
    arguments = [(nodes[146], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 147. fused_subtract42
    # kernel 0
    output_size = 1572864
    nodes[147] = kaas.bufferSpec('147', output_size, const=False, ephemeral=True)
    arguments = [(nodes[147], 'o'), (nodes[145], 'i'), (nodes[146], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 148. fused_power_mean42
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a13', output_size, const=False, ephemeral=True))
    arguments = [(nodes[147], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[148] = kaas.bufferSpec('148', output_size, const=False, ephemeral=True)
    arguments = [(nodes[148], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 149. p51
    nodes[149] = addToKV(149, params['p51'])

    # 150. p52
    nodes[150] = addToKV(150, params['p52'])

    # 151. fused_add_sqrt_divide_multiply_add41
    # kernel 0
    output_size = 1572864
    nodes[151] = kaas.bufferSpec('151', output_size, const=False, ephemeral=True)
    arguments = [(nodes[151], 'o'), (nodes[148], 'i'), (nodes[147], 'i'), (nodes[149], 'i'), (nodes[150], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 152. reshape_nop
    nodes[152] = nodes[151]

    # 153. p53
    nodes[153] = addToKV(153, params['p53'])

    # 154. fused_nn_batch_matmul_341
    # kernel 0
    output_size = 1572864
    nodes[154] = kaas.bufferSpec('154', output_size, const=False, ephemeral=True)
    arguments = [(nodes[152], 'i'), (nodes[153], 'i'), (nodes[154], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 155. p54
    nodes[155] = addToKV(155, params['p54'])

    # 156. fused_reshape_add_reshape_transpose_reshape20
    # kernel 0
    output_size = 1572864
    nodes[156] = kaas.bufferSpec('156', output_size, const=False, ephemeral=True)
    arguments = [(nodes[156], 'o'), (nodes[154], 'i'), (nodes[155], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 157. p55
    nodes[157] = addToKV(157, params['p55'])

    # 158. fused_nn_batch_matmul_354
    # kernel 0
    output_size = 1572864
    nodes[158] = kaas.bufferSpec('158', output_size, const=False, ephemeral=True)
    arguments = [(nodes[152], 'i'), (nodes[157], 'i'), (nodes[158], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 159. p56
    nodes[159] = addToKV(159, params['p56'])

    # 160. fused_reshape_add_reshape_transpose_reshape_transpose3
    # kernel 0
    output_size = 1572864
    nodes[160] = kaas.bufferSpec('160', output_size, const=False, ephemeral=True)
    arguments = [(nodes[160], 'o'), (nodes[158], 'i'), (nodes[159], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 161. fused_nn_batch_matmul_520
    # kernel 0
    output_size = 9437184
    nodes[161] = kaas.bufferSpec('161', output_size, const=False, ephemeral=True)
    arguments = [(nodes[156], 'i'), (nodes[160], 'i'), (nodes[161], 'o'), ]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 162. fused_reshape_divide_add20
    # kernel 0
    output_size = 9437184
    nodes[162] = kaas.bufferSpec('162', output_size, const=False, ephemeral=True)
    arguments = [(nodes[162], 'o'), (nodes[161], 'i'), (nodes[23], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 163. fused_max3
    # kernel 0
    output_size = 24576
    nodes[163] = kaas.bufferSpec('163', output_size, const=False, ephemeral=True)
    arguments = [(nodes[162], 'i'), (nodes[163], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 164. fused_subtract_exp20
    # kernel 0
    output_size = 9437184
    nodes[164] = kaas.bufferSpec('164', output_size, const=False, ephemeral=True)
    arguments = [(nodes[164], 'o'), (nodes[162], 'i'), (nodes[163], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 165. fused_sum3
    # kernel 0
    output_size = 24576
    nodes[165] = kaas.bufferSpec('165', output_size, const=False, ephemeral=True)
    arguments = [(nodes[164], 'i'), (nodes[165], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 166. fused_divide_reshape20
    # kernel 0
    output_size = 9437184
    nodes[166] = kaas.bufferSpec('166', output_size, const=False, ephemeral=True)
    arguments = [(nodes[166], 'o'), (nodes[164], 'i'), (nodes[165], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 167. p57
    nodes[167] = addToKV(167, params['p57'])

    # 168. fused_nn_batch_matmul_355
    # kernel 0
    output_size = 1572864
    nodes[168] = kaas.bufferSpec('168', output_size, const=False, ephemeral=True)
    arguments = [(nodes[152], 'i'), (nodes[167], 'i'), (nodes[168], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 169. p58
    nodes[169] = addToKV(169, params['p58'])

    # 170. fused_reshape_add_reshape_transpose_reshape_transpose_13
    # kernel 0
    output_size = 1572864
    nodes[170] = kaas.bufferSpec('170', output_size, const=False, ephemeral=True)
    arguments = [(nodes[170], 'o'), (nodes[168], 'i'), (nodes[169], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 171. fused_nn_batch_matmul_420
    # kernel 0
    output_size = 1572864
    nodes[171] = kaas.bufferSpec('171', output_size, const=False, ephemeral=True)
    arguments = [(nodes[166], 'i'), (nodes[170], 'i'), (nodes[171], 'o'), ]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 172. fused_reshape_transpose_reshape20
    # kernel 0
    output_size = 1572864
    nodes[172] = kaas.bufferSpec('172', output_size, const=False, ephemeral=True)
    arguments = [(nodes[172], 'o'), (nodes[171], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 173. p59
    nodes[173] = addToKV(173, params['p59'])

    # 174. fused_nn_batch_matmul_340
    # kernel 0
    output_size = 1572864
    nodes[174] = kaas.bufferSpec('174', output_size, const=False, ephemeral=True)
    arguments = [(nodes[172], 'i'), (nodes[173], 'i'), (nodes[174], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 175. p60
    nodes[175] = addToKV(175, params['p60'])

    # 176. fused_reshape_add_add41
    # kernel 0
    output_size = 1572864
    nodes[176] = kaas.bufferSpec('176', output_size, const=False, ephemeral=True)
    arguments = [(nodes[176], 'o'), (nodes[174], 'i'), (nodes[175], 'i'), (nodes[151], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 177. fused_mean7
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a14', output_size, const=False, ephemeral=True))
    arguments = [(nodes[176], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[177] = kaas.bufferSpec('177', output_size, const=False, ephemeral=True)
    arguments = [(nodes[177], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 178. fused_subtract41
    # kernel 0
    output_size = 1572864
    nodes[178] = kaas.bufferSpec('178', output_size, const=False, ephemeral=True)
    arguments = [(nodes[178], 'o'), (nodes[176], 'i'), (nodes[177], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 179. fused_power_mean41
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a15', output_size, const=False, ephemeral=True))
    arguments = [(nodes[178], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[179] = kaas.bufferSpec('179', output_size, const=False, ephemeral=True)
    arguments = [(nodes[179], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 180. p61
    nodes[180] = addToKV(180, params['p61'])

    # 181. p62
    nodes[181] = addToKV(181, params['p62'])

    # 182. fused_add_sqrt_divide_multiply_add40
    # kernel 0
    output_size = 1572864
    nodes[182] = kaas.bufferSpec('182', output_size, const=False, ephemeral=True)
    arguments = [(nodes[182], 'o'), (nodes[179], 'i'), (nodes[178], 'i'), (nodes[180], 'i'), (nodes[181], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 183. reshape_nop
    nodes[183] = nodes[182]

    # 184. p63
    nodes[184] = addToKV(184, params['p63'])

    # 185. fused_nn_batch_matmul_220
    # kernel 0
    output_size = 6291456
    nodes[185] = kaas.bufferSpec('185', output_size, const=False, ephemeral=True)
    arguments = [(nodes[183], 'i'), (nodes[184], 'i'), (nodes[185], 'o'), ]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 186. p64
    nodes[186] = addToKV(186, params['p64'])

    # 187. fused_reshape_add_multiply_divide_erf_add_multiply_reshape20
    # kernel 0
    output_size = 6291456
    nodes[187] = kaas.bufferSpec('187', output_size, const=False, ephemeral=True)
    arguments = [(nodes[187], 'o'), (nodes[185], 'i'), (nodes[186], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 188. p65
    nodes[188] = addToKV(188, params['p65'])

    # 189. fused_nn_batch_matmul_120
    # kernel 0
    output_size = 1572864
    nodes[189] = kaas.bufferSpec('189', output_size, const=False, ephemeral=True)
    arguments = [(nodes[187], 'i'), (nodes[188], 'i'), (nodes[189], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 190. p66
    nodes[190] = addToKV(190, params['p66'])

    # 191. fused_reshape_add_add40
    # kernel 0
    output_size = 1572864
    nodes[191] = kaas.bufferSpec('191', output_size, const=False, ephemeral=True)
    arguments = [(nodes[191], 'o'), (nodes[189], 'i'), (nodes[190], 'i'), (nodes[182], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 192. fused_mean8
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a16', output_size, const=False, ephemeral=True))
    arguments = [(nodes[191], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[192] = kaas.bufferSpec('192', output_size, const=False, ephemeral=True)
    arguments = [(nodes[192], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 193. fused_subtract40
    # kernel 0
    output_size = 1572864
    nodes[193] = kaas.bufferSpec('193', output_size, const=False, ephemeral=True)
    arguments = [(nodes[193], 'o'), (nodes[191], 'i'), (nodes[192], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 194. fused_power_mean40
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a17', output_size, const=False, ephemeral=True))
    arguments = [(nodes[193], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[194] = kaas.bufferSpec('194', output_size, const=False, ephemeral=True)
    arguments = [(nodes[194], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 195. p67
    nodes[195] = addToKV(195, params['p67'])

    # 196. p68
    nodes[196] = addToKV(196, params['p68'])

    # 197. fused_add_sqrt_divide_multiply_add39
    # kernel 0
    output_size = 1572864
    nodes[197] = kaas.bufferSpec('197', output_size, const=False, ephemeral=True)
    arguments = [(nodes[197], 'o'), (nodes[194], 'i'), (nodes[193], 'i'), (nodes[195], 'i'), (nodes[196], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 198. reshape_nop
    nodes[198] = nodes[197]

    # 199. p69
    nodes[199] = addToKV(199, params['p69'])

    # 200. fused_nn_batch_matmul_339
    # kernel 0
    output_size = 1572864
    nodes[200] = kaas.bufferSpec('200', output_size, const=False, ephemeral=True)
    arguments = [(nodes[198], 'i'), (nodes[199], 'i'), (nodes[200], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 201. p70
    nodes[201] = addToKV(201, params['p70'])

    # 202. fused_reshape_add_reshape_transpose_reshape19
    # kernel 0
    output_size = 1572864
    nodes[202] = kaas.bufferSpec('202', output_size, const=False, ephemeral=True)
    arguments = [(nodes[202], 'o'), (nodes[200], 'i'), (nodes[201], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 203. p71
    nodes[203] = addToKV(203, params['p71'])

    # 204. fused_nn_batch_matmul_356
    # kernel 0
    output_size = 1572864
    nodes[204] = kaas.bufferSpec('204', output_size, const=False, ephemeral=True)
    arguments = [(nodes[198], 'i'), (nodes[203], 'i'), (nodes[204], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 205. p72
    nodes[205] = addToKV(205, params['p72'])

    # 206. fused_reshape_add_reshape_transpose_reshape_transpose4
    # kernel 0
    output_size = 1572864
    nodes[206] = kaas.bufferSpec('206', output_size, const=False, ephemeral=True)
    arguments = [(nodes[206], 'o'), (nodes[204], 'i'), (nodes[205], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 207. fused_nn_batch_matmul_519
    # kernel 0
    output_size = 9437184
    nodes[207] = kaas.bufferSpec('207', output_size, const=False, ephemeral=True)
    arguments = [(nodes[202], 'i'), (nodes[206], 'i'), (nodes[207], 'o'), ]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 208. fused_reshape_divide_add19
    # kernel 0
    output_size = 9437184
    nodes[208] = kaas.bufferSpec('208', output_size, const=False, ephemeral=True)
    arguments = [(nodes[208], 'o'), (nodes[207], 'i'), (nodes[23], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 209. fused_max4
    # kernel 0
    output_size = 24576
    nodes[209] = kaas.bufferSpec('209', output_size, const=False, ephemeral=True)
    arguments = [(nodes[208], 'i'), (nodes[209], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 210. fused_subtract_exp19
    # kernel 0
    output_size = 9437184
    nodes[210] = kaas.bufferSpec('210', output_size, const=False, ephemeral=True)
    arguments = [(nodes[210], 'o'), (nodes[208], 'i'), (nodes[209], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 211. fused_sum4
    # kernel 0
    output_size = 24576
    nodes[211] = kaas.bufferSpec('211', output_size, const=False, ephemeral=True)
    arguments = [(nodes[210], 'i'), (nodes[211], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 212. fused_divide_reshape19
    # kernel 0
    output_size = 9437184
    nodes[212] = kaas.bufferSpec('212', output_size, const=False, ephemeral=True)
    arguments = [(nodes[212], 'o'), (nodes[210], 'i'), (nodes[211], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 213. p73
    nodes[213] = addToKV(213, params['p73'])

    # 214. fused_nn_batch_matmul_357
    # kernel 0
    output_size = 1572864
    nodes[214] = kaas.bufferSpec('214', output_size, const=False, ephemeral=True)
    arguments = [(nodes[198], 'i'), (nodes[213], 'i'), (nodes[214], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 215. p74
    nodes[215] = addToKV(215, params['p74'])

    # 216. fused_reshape_add_reshape_transpose_reshape_transpose_14
    # kernel 0
    output_size = 1572864
    nodes[216] = kaas.bufferSpec('216', output_size, const=False, ephemeral=True)
    arguments = [(nodes[216], 'o'), (nodes[214], 'i'), (nodes[215], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 217. fused_nn_batch_matmul_419
    # kernel 0
    output_size = 1572864
    nodes[217] = kaas.bufferSpec('217', output_size, const=False, ephemeral=True)
    arguments = [(nodes[212], 'i'), (nodes[216], 'i'), (nodes[217], 'o'), ]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 218. fused_reshape_transpose_reshape19
    # kernel 0
    output_size = 1572864
    nodes[218] = kaas.bufferSpec('218', output_size, const=False, ephemeral=True)
    arguments = [(nodes[218], 'o'), (nodes[217], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 219. p75
    nodes[219] = addToKV(219, params['p75'])

    # 220. fused_nn_batch_matmul_338
    # kernel 0
    output_size = 1572864
    nodes[220] = kaas.bufferSpec('220', output_size, const=False, ephemeral=True)
    arguments = [(nodes[218], 'i'), (nodes[219], 'i'), (nodes[220], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 221. p76
    nodes[221] = addToKV(221, params['p76'])

    # 222. fused_reshape_add_add39
    # kernel 0
    output_size = 1572864
    nodes[222] = kaas.bufferSpec('222', output_size, const=False, ephemeral=True)
    arguments = [(nodes[222], 'o'), (nodes[220], 'i'), (nodes[221], 'i'), (nodes[197], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 223. fused_mean9
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a18', output_size, const=False, ephemeral=True))
    arguments = [(nodes[222], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[223] = kaas.bufferSpec('223', output_size, const=False, ephemeral=True)
    arguments = [(nodes[223], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 224. fused_subtract39
    # kernel 0
    output_size = 1572864
    nodes[224] = kaas.bufferSpec('224', output_size, const=False, ephemeral=True)
    arguments = [(nodes[224], 'o'), (nodes[222], 'i'), (nodes[223], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 225. fused_power_mean39
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a19', output_size, const=False, ephemeral=True))
    arguments = [(nodes[224], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[225] = kaas.bufferSpec('225', output_size, const=False, ephemeral=True)
    arguments = [(nodes[225], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 226. p77
    nodes[226] = addToKV(226, params['p77'])

    # 227. p78
    nodes[227] = addToKV(227, params['p78'])

    # 228. fused_add_sqrt_divide_multiply_add38
    # kernel 0
    output_size = 1572864
    nodes[228] = kaas.bufferSpec('228', output_size, const=False, ephemeral=True)
    arguments = [(nodes[228], 'o'), (nodes[225], 'i'), (nodes[224], 'i'), (nodes[226], 'i'), (nodes[227], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 229. reshape_nop
    nodes[229] = nodes[228]

    # 230. p79
    nodes[230] = addToKV(230, params['p79'])

    # 231. fused_nn_batch_matmul_219
    # kernel 0
    output_size = 6291456
    nodes[231] = kaas.bufferSpec('231', output_size, const=False, ephemeral=True)
    arguments = [(nodes[229], 'i'), (nodes[230], 'i'), (nodes[231], 'o'), ]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 232. p80
    nodes[232] = addToKV(232, params['p80'])

    # 233. fused_reshape_add_multiply_divide_erf_add_multiply_reshape19
    # kernel 0
    output_size = 6291456
    nodes[233] = kaas.bufferSpec('233', output_size, const=False, ephemeral=True)
    arguments = [(nodes[233], 'o'), (nodes[231], 'i'), (nodes[232], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 234. p81
    nodes[234] = addToKV(234, params['p81'])

    # 235. fused_nn_batch_matmul_119
    # kernel 0
    output_size = 1572864
    nodes[235] = kaas.bufferSpec('235', output_size, const=False, ephemeral=True)
    arguments = [(nodes[233], 'i'), (nodes[234], 'i'), (nodes[235], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 236. p82
    nodes[236] = addToKV(236, params['p82'])

    # 237. fused_reshape_add_add38
    # kernel 0
    output_size = 1572864
    nodes[237] = kaas.bufferSpec('237', output_size, const=False, ephemeral=True)
    arguments = [(nodes[237], 'o'), (nodes[235], 'i'), (nodes[236], 'i'), (nodes[228], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 238. fused_mean10
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a20', output_size, const=False, ephemeral=True))
    arguments = [(nodes[237], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[238] = kaas.bufferSpec('238', output_size, const=False, ephemeral=True)
    arguments = [(nodes[238], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 239. fused_subtract38
    # kernel 0
    output_size = 1572864
    nodes[239] = kaas.bufferSpec('239', output_size, const=False, ephemeral=True)
    arguments = [(nodes[239], 'o'), (nodes[237], 'i'), (nodes[238], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 240. fused_power_mean38
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a21', output_size, const=False, ephemeral=True))
    arguments = [(nodes[239], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[240] = kaas.bufferSpec('240', output_size, const=False, ephemeral=True)
    arguments = [(nodes[240], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 241. p83
    nodes[241] = addToKV(241, params['p83'])

    # 242. p84
    nodes[242] = addToKV(242, params['p84'])

    # 243. fused_add_sqrt_divide_multiply_add37
    # kernel 0
    output_size = 1572864
    nodes[243] = kaas.bufferSpec('243', output_size, const=False, ephemeral=True)
    arguments = [(nodes[243], 'o'), (nodes[240], 'i'), (nodes[239], 'i'), (nodes[241], 'i'), (nodes[242], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 244. reshape_nop
    nodes[244] = nodes[243]

    # 245. p85
    nodes[245] = addToKV(245, params['p85'])

    # 246. fused_nn_batch_matmul_337
    # kernel 0
    output_size = 1572864
    nodes[246] = kaas.bufferSpec('246', output_size, const=False, ephemeral=True)
    arguments = [(nodes[244], 'i'), (nodes[245], 'i'), (nodes[246], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 247. p86
    nodes[247] = addToKV(247, params['p86'])

    # 248. fused_reshape_add_reshape_transpose_reshape18
    # kernel 0
    output_size = 1572864
    nodes[248] = kaas.bufferSpec('248', output_size, const=False, ephemeral=True)
    arguments = [(nodes[248], 'o'), (nodes[246], 'i'), (nodes[247], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 249. p87
    nodes[249] = addToKV(249, params['p87'])

    # 250. fused_nn_batch_matmul_358
    # kernel 0
    output_size = 1572864
    nodes[250] = kaas.bufferSpec('250', output_size, const=False, ephemeral=True)
    arguments = [(nodes[244], 'i'), (nodes[249], 'i'), (nodes[250], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 251. p88
    nodes[251] = addToKV(251, params['p88'])

    # 252. fused_reshape_add_reshape_transpose_reshape_transpose5
    # kernel 0
    output_size = 1572864
    nodes[252] = kaas.bufferSpec('252', output_size, const=False, ephemeral=True)
    arguments = [(nodes[252], 'o'), (nodes[250], 'i'), (nodes[251], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 253. fused_nn_batch_matmul_518
    # kernel 0
    output_size = 9437184
    nodes[253] = kaas.bufferSpec('253', output_size, const=False, ephemeral=True)
    arguments = [(nodes[248], 'i'), (nodes[252], 'i'), (nodes[253], 'o'), ]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 254. fused_reshape_divide_add18
    # kernel 0
    output_size = 9437184
    nodes[254] = kaas.bufferSpec('254', output_size, const=False, ephemeral=True)
    arguments = [(nodes[254], 'o'), (nodes[253], 'i'), (nodes[23], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 255. fused_max5
    # kernel 0
    output_size = 24576
    nodes[255] = kaas.bufferSpec('255', output_size, const=False, ephemeral=True)
    arguments = [(nodes[254], 'i'), (nodes[255], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 256. fused_subtract_exp18
    # kernel 0
    output_size = 9437184
    nodes[256] = kaas.bufferSpec('256', output_size, const=False, ephemeral=True)
    arguments = [(nodes[256], 'o'), (nodes[254], 'i'), (nodes[255], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 257. fused_sum5
    # kernel 0
    output_size = 24576
    nodes[257] = kaas.bufferSpec('257', output_size, const=False, ephemeral=True)
    arguments = [(nodes[256], 'i'), (nodes[257], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 258. fused_divide_reshape18
    # kernel 0
    output_size = 9437184
    nodes[258] = kaas.bufferSpec('258', output_size, const=False, ephemeral=True)
    arguments = [(nodes[258], 'o'), (nodes[256], 'i'), (nodes[257], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 259. p89
    nodes[259] = addToKV(259, params['p89'])

    # 260. fused_nn_batch_matmul_359
    # kernel 0
    output_size = 1572864
    nodes[260] = kaas.bufferSpec('260', output_size, const=False, ephemeral=True)
    arguments = [(nodes[244], 'i'), (nodes[259], 'i'), (nodes[260], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 261. p90
    nodes[261] = addToKV(261, params['p90'])

    # 262. fused_reshape_add_reshape_transpose_reshape_transpose_15
    # kernel 0
    output_size = 1572864
    nodes[262] = kaas.bufferSpec('262', output_size, const=False, ephemeral=True)
    arguments = [(nodes[262], 'o'), (nodes[260], 'i'), (nodes[261], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 263. fused_nn_batch_matmul_418
    # kernel 0
    output_size = 1572864
    nodes[263] = kaas.bufferSpec('263', output_size, const=False, ephemeral=True)
    arguments = [(nodes[258], 'i'), (nodes[262], 'i'), (nodes[263], 'o'), ]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 264. fused_reshape_transpose_reshape18
    # kernel 0
    output_size = 1572864
    nodes[264] = kaas.bufferSpec('264', output_size, const=False, ephemeral=True)
    arguments = [(nodes[264], 'o'), (nodes[263], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 265. p91
    nodes[265] = addToKV(265, params['p91'])

    # 266. fused_nn_batch_matmul_336
    # kernel 0
    output_size = 1572864
    nodes[266] = kaas.bufferSpec('266', output_size, const=False, ephemeral=True)
    arguments = [(nodes[264], 'i'), (nodes[265], 'i'), (nodes[266], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 267. p92
    nodes[267] = addToKV(267, params['p92'])

    # 268. fused_reshape_add_add37
    # kernel 0
    output_size = 1572864
    nodes[268] = kaas.bufferSpec('268', output_size, const=False, ephemeral=True)
    arguments = [(nodes[268], 'o'), (nodes[266], 'i'), (nodes[267], 'i'), (nodes[243], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 269. fused_mean11
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a22', output_size, const=False, ephemeral=True))
    arguments = [(nodes[268], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[269] = kaas.bufferSpec('269', output_size, const=False, ephemeral=True)
    arguments = [(nodes[269], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 270. fused_subtract37
    # kernel 0
    output_size = 1572864
    nodes[270] = kaas.bufferSpec('270', output_size, const=False, ephemeral=True)
    arguments = [(nodes[270], 'o'), (nodes[268], 'i'), (nodes[269], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 271. fused_power_mean37
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a23', output_size, const=False, ephemeral=True))
    arguments = [(nodes[270], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[271] = kaas.bufferSpec('271', output_size, const=False, ephemeral=True)
    arguments = [(nodes[271], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 272. p93
    nodes[272] = addToKV(272, params['p93'])

    # 273. p94
    nodes[273] = addToKV(273, params['p94'])

    # 274. fused_add_sqrt_divide_multiply_add36
    # kernel 0
    output_size = 1572864
    nodes[274] = kaas.bufferSpec('274', output_size, const=False, ephemeral=True)
    arguments = [(nodes[274], 'o'), (nodes[271], 'i'), (nodes[270], 'i'), (nodes[272], 'i'), (nodes[273], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 275. reshape_nop
    nodes[275] = nodes[274]

    # 276. p95
    nodes[276] = addToKV(276, params['p95'])

    # 277. fused_nn_batch_matmul_218
    # kernel 0
    output_size = 6291456
    nodes[277] = kaas.bufferSpec('277', output_size, const=False, ephemeral=True)
    arguments = [(nodes[275], 'i'), (nodes[276], 'i'), (nodes[277], 'o'), ]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 278. p96
    nodes[278] = addToKV(278, params['p96'])

    # 279. fused_reshape_add_multiply_divide_erf_add_multiply_reshape18
    # kernel 0
    output_size = 6291456
    nodes[279] = kaas.bufferSpec('279', output_size, const=False, ephemeral=True)
    arguments = [(nodes[279], 'o'), (nodes[277], 'i'), (nodes[278], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 280. p97
    nodes[280] = addToKV(280, params['p97'])

    # 281. fused_nn_batch_matmul_118
    # kernel 0
    output_size = 1572864
    nodes[281] = kaas.bufferSpec('281', output_size, const=False, ephemeral=True)
    arguments = [(nodes[279], 'i'), (nodes[280], 'i'), (nodes[281], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 282. p98
    nodes[282] = addToKV(282, params['p98'])

    # 283. fused_reshape_add_add36
    # kernel 0
    output_size = 1572864
    nodes[283] = kaas.bufferSpec('283', output_size, const=False, ephemeral=True)
    arguments = [(nodes[283], 'o'), (nodes[281], 'i'), (nodes[282], 'i'), (nodes[274], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 284. fused_mean12
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a24', output_size, const=False, ephemeral=True))
    arguments = [(nodes[283], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[284] = kaas.bufferSpec('284', output_size, const=False, ephemeral=True)
    arguments = [(nodes[284], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 285. fused_subtract36
    # kernel 0
    output_size = 1572864
    nodes[285] = kaas.bufferSpec('285', output_size, const=False, ephemeral=True)
    arguments = [(nodes[285], 'o'), (nodes[283], 'i'), (nodes[284], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 286. fused_power_mean36
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a25', output_size, const=False, ephemeral=True))
    arguments = [(nodes[285], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[286] = kaas.bufferSpec('286', output_size, const=False, ephemeral=True)
    arguments = [(nodes[286], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 287. p99
    nodes[287] = addToKV(287, params['p99'])

    # 288. p100
    nodes[288] = addToKV(288, params['p100'])

    # 289. fused_add_sqrt_divide_multiply_add35
    # kernel 0
    output_size = 1572864
    nodes[289] = kaas.bufferSpec('289', output_size, const=False, ephemeral=True)
    arguments = [(nodes[289], 'o'), (nodes[286], 'i'), (nodes[285], 'i'), (nodes[287], 'i'), (nodes[288], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 290. reshape_nop
    nodes[290] = nodes[289]

    # 291. p101
    nodes[291] = addToKV(291, params['p101'])

    # 292. fused_nn_batch_matmul_335
    # kernel 0
    output_size = 1572864
    nodes[292] = kaas.bufferSpec('292', output_size, const=False, ephemeral=True)
    arguments = [(nodes[290], 'i'), (nodes[291], 'i'), (nodes[292], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 293. p102
    nodes[293] = addToKV(293, params['p102'])

    # 294. fused_reshape_add_reshape_transpose_reshape17
    # kernel 0
    output_size = 1572864
    nodes[294] = kaas.bufferSpec('294', output_size, const=False, ephemeral=True)
    arguments = [(nodes[294], 'o'), (nodes[292], 'i'), (nodes[293], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 295. p103
    nodes[295] = addToKV(295, params['p103'])

    # 296. fused_nn_batch_matmul_360
    # kernel 0
    output_size = 1572864
    nodes[296] = kaas.bufferSpec('296', output_size, const=False, ephemeral=True)
    arguments = [(nodes[290], 'i'), (nodes[295], 'i'), (nodes[296], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 297. p104
    nodes[297] = addToKV(297, params['p104'])

    # 298. fused_reshape_add_reshape_transpose_reshape_transpose6
    # kernel 0
    output_size = 1572864
    nodes[298] = kaas.bufferSpec('298', output_size, const=False, ephemeral=True)
    arguments = [(nodes[298], 'o'), (nodes[296], 'i'), (nodes[297], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 299. fused_nn_batch_matmul_517
    # kernel 0
    output_size = 9437184
    nodes[299] = kaas.bufferSpec('299', output_size, const=False, ephemeral=True)
    arguments = [(nodes[294], 'i'), (nodes[298], 'i'), (nodes[299], 'o'), ]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 300. fused_reshape_divide_add17
    # kernel 0
    output_size = 9437184
    nodes[300] = kaas.bufferSpec('300', output_size, const=False, ephemeral=True)
    arguments = [(nodes[300], 'o'), (nodes[299], 'i'), (nodes[23], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 301. fused_max6
    # kernel 0
    output_size = 24576
    nodes[301] = kaas.bufferSpec('301', output_size, const=False, ephemeral=True)
    arguments = [(nodes[300], 'i'), (nodes[301], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 302. fused_subtract_exp17
    # kernel 0
    output_size = 9437184
    nodes[302] = kaas.bufferSpec('302', output_size, const=False, ephemeral=True)
    arguments = [(nodes[302], 'o'), (nodes[300], 'i'), (nodes[301], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 303. fused_sum6
    # kernel 0
    output_size = 24576
    nodes[303] = kaas.bufferSpec('303', output_size, const=False, ephemeral=True)
    arguments = [(nodes[302], 'i'), (nodes[303], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 304. fused_divide_reshape17
    # kernel 0
    output_size = 9437184
    nodes[304] = kaas.bufferSpec('304', output_size, const=False, ephemeral=True)
    arguments = [(nodes[304], 'o'), (nodes[302], 'i'), (nodes[303], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 305. p105
    nodes[305] = addToKV(305, params['p105'])

    # 306. fused_nn_batch_matmul_361
    # kernel 0
    output_size = 1572864
    nodes[306] = kaas.bufferSpec('306', output_size, const=False, ephemeral=True)
    arguments = [(nodes[290], 'i'), (nodes[305], 'i'), (nodes[306], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 307. p106
    nodes[307] = addToKV(307, params['p106'])

    # 308. fused_reshape_add_reshape_transpose_reshape_transpose_16
    # kernel 0
    output_size = 1572864
    nodes[308] = kaas.bufferSpec('308', output_size, const=False, ephemeral=True)
    arguments = [(nodes[308], 'o'), (nodes[306], 'i'), (nodes[307], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 309. fused_nn_batch_matmul_417
    # kernel 0
    output_size = 1572864
    nodes[309] = kaas.bufferSpec('309', output_size, const=False, ephemeral=True)
    arguments = [(nodes[304], 'i'), (nodes[308], 'i'), (nodes[309], 'o'), ]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 310. fused_reshape_transpose_reshape17
    # kernel 0
    output_size = 1572864
    nodes[310] = kaas.bufferSpec('310', output_size, const=False, ephemeral=True)
    arguments = [(nodes[310], 'o'), (nodes[309], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 311. p107
    nodes[311] = addToKV(311, params['p107'])

    # 312. fused_nn_batch_matmul_334
    # kernel 0
    output_size = 1572864
    nodes[312] = kaas.bufferSpec('312', output_size, const=False, ephemeral=True)
    arguments = [(nodes[310], 'i'), (nodes[311], 'i'), (nodes[312], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 313. p108
    nodes[313] = addToKV(313, params['p108'])

    # 314. fused_reshape_add_add35
    # kernel 0
    output_size = 1572864
    nodes[314] = kaas.bufferSpec('314', output_size, const=False, ephemeral=True)
    arguments = [(nodes[314], 'o'), (nodes[312], 'i'), (nodes[313], 'i'), (nodes[289], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 315. fused_mean13
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a26', output_size, const=False, ephemeral=True))
    arguments = [(nodes[314], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[315] = kaas.bufferSpec('315', output_size, const=False, ephemeral=True)
    arguments = [(nodes[315], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 316. fused_subtract35
    # kernel 0
    output_size = 1572864
    nodes[316] = kaas.bufferSpec('316', output_size, const=False, ephemeral=True)
    arguments = [(nodes[316], 'o'), (nodes[314], 'i'), (nodes[315], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 317. fused_power_mean35
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a27', output_size, const=False, ephemeral=True))
    arguments = [(nodes[316], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[317] = kaas.bufferSpec('317', output_size, const=False, ephemeral=True)
    arguments = [(nodes[317], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 318. p109
    nodes[318] = addToKV(318, params['p109'])

    # 319. p110
    nodes[319] = addToKV(319, params['p110'])

    # 320. fused_add_sqrt_divide_multiply_add34
    # kernel 0
    output_size = 1572864
    nodes[320] = kaas.bufferSpec('320', output_size, const=False, ephemeral=True)
    arguments = [(nodes[320], 'o'), (nodes[317], 'i'), (nodes[316], 'i'), (nodes[318], 'i'), (nodes[319], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 321. reshape_nop
    nodes[321] = nodes[320]

    # 322. p111
    nodes[322] = addToKV(322, params['p111'])

    # 323. fused_nn_batch_matmul_217
    # kernel 0
    output_size = 6291456
    nodes[323] = kaas.bufferSpec('323', output_size, const=False, ephemeral=True)
    arguments = [(nodes[321], 'i'), (nodes[322], 'i'), (nodes[323], 'o'), ]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 324. p112
    nodes[324] = addToKV(324, params['p112'])

    # 325. fused_reshape_add_multiply_divide_erf_add_multiply_reshape17
    # kernel 0
    output_size = 6291456
    nodes[325] = kaas.bufferSpec('325', output_size, const=False, ephemeral=True)
    arguments = [(nodes[325], 'o'), (nodes[323], 'i'), (nodes[324], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 326. p113
    nodes[326] = addToKV(326, params['p113'])

    # 327. fused_nn_batch_matmul_117
    # kernel 0
    output_size = 1572864
    nodes[327] = kaas.bufferSpec('327', output_size, const=False, ephemeral=True)
    arguments = [(nodes[325], 'i'), (nodes[326], 'i'), (nodes[327], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 328. p114
    nodes[328] = addToKV(328, params['p114'])

    # 329. fused_reshape_add_add34
    # kernel 0
    output_size = 1572864
    nodes[329] = kaas.bufferSpec('329', output_size, const=False, ephemeral=True)
    arguments = [(nodes[329], 'o'), (nodes[327], 'i'), (nodes[328], 'i'), (nodes[320], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 330. fused_mean14
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a28', output_size, const=False, ephemeral=True))
    arguments = [(nodes[329], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[330] = kaas.bufferSpec('330', output_size, const=False, ephemeral=True)
    arguments = [(nodes[330], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 331. fused_subtract34
    # kernel 0
    output_size = 1572864
    nodes[331] = kaas.bufferSpec('331', output_size, const=False, ephemeral=True)
    arguments = [(nodes[331], 'o'), (nodes[329], 'i'), (nodes[330], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 332. fused_power_mean34
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a29', output_size, const=False, ephemeral=True))
    arguments = [(nodes[331], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[332] = kaas.bufferSpec('332', output_size, const=False, ephemeral=True)
    arguments = [(nodes[332], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 333. p115
    nodes[333] = addToKV(333, params['p115'])

    # 334. p116
    nodes[334] = addToKV(334, params['p116'])

    # 335. fused_add_sqrt_divide_multiply_add33
    # kernel 0
    output_size = 1572864
    nodes[335] = kaas.bufferSpec('335', output_size, const=False, ephemeral=True)
    arguments = [(nodes[335], 'o'), (nodes[332], 'i'), (nodes[331], 'i'), (nodes[333], 'i'), (nodes[334], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 336. reshape_nop
    nodes[336] = nodes[335]

    # 337. p117
    nodes[337] = addToKV(337, params['p117'])

    # 338. fused_nn_batch_matmul_333
    # kernel 0
    output_size = 1572864
    nodes[338] = kaas.bufferSpec('338', output_size, const=False, ephemeral=True)
    arguments = [(nodes[336], 'i'), (nodes[337], 'i'), (nodes[338], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 339. p118
    nodes[339] = addToKV(339, params['p118'])

    # 340. fused_reshape_add_reshape_transpose_reshape16
    # kernel 0
    output_size = 1572864
    nodes[340] = kaas.bufferSpec('340', output_size, const=False, ephemeral=True)
    arguments = [(nodes[340], 'o'), (nodes[338], 'i'), (nodes[339], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 341. p119
    nodes[341] = addToKV(341, params['p119'])

    # 342. fused_nn_batch_matmul_362
    # kernel 0
    output_size = 1572864
    nodes[342] = kaas.bufferSpec('342', output_size, const=False, ephemeral=True)
    arguments = [(nodes[336], 'i'), (nodes[341], 'i'), (nodes[342], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 343. p120
    nodes[343] = addToKV(343, params['p120'])

    # 344. fused_reshape_add_reshape_transpose_reshape_transpose7
    # kernel 0
    output_size = 1572864
    nodes[344] = kaas.bufferSpec('344', output_size, const=False, ephemeral=True)
    arguments = [(nodes[344], 'o'), (nodes[342], 'i'), (nodes[343], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 345. fused_nn_batch_matmul_516
    # kernel 0
    output_size = 9437184
    nodes[345] = kaas.bufferSpec('345', output_size, const=False, ephemeral=True)
    arguments = [(nodes[340], 'i'), (nodes[344], 'i'), (nodes[345], 'o'), ]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 346. fused_reshape_divide_add16
    # kernel 0
    output_size = 9437184
    nodes[346] = kaas.bufferSpec('346', output_size, const=False, ephemeral=True)
    arguments = [(nodes[346], 'o'), (nodes[345], 'i'), (nodes[23], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 347. fused_max7
    # kernel 0
    output_size = 24576
    nodes[347] = kaas.bufferSpec('347', output_size, const=False, ephemeral=True)
    arguments = [(nodes[346], 'i'), (nodes[347], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 348. fused_subtract_exp16
    # kernel 0
    output_size = 9437184
    nodes[348] = kaas.bufferSpec('348', output_size, const=False, ephemeral=True)
    arguments = [(nodes[348], 'o'), (nodes[346], 'i'), (nodes[347], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 349. fused_sum7
    # kernel 0
    output_size = 24576
    nodes[349] = kaas.bufferSpec('349', output_size, const=False, ephemeral=True)
    arguments = [(nodes[348], 'i'), (nodes[349], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 350. fused_divide_reshape16
    # kernel 0
    output_size = 9437184
    nodes[350] = kaas.bufferSpec('350', output_size, const=False, ephemeral=True)
    arguments = [(nodes[350], 'o'), (nodes[348], 'i'), (nodes[349], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 351. p121
    nodes[351] = addToKV(351, params['p121'])

    # 352. fused_nn_batch_matmul_363
    # kernel 0
    output_size = 1572864
    nodes[352] = kaas.bufferSpec('352', output_size, const=False, ephemeral=True)
    arguments = [(nodes[336], 'i'), (nodes[351], 'i'), (nodes[352], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 353. p122
    nodes[353] = addToKV(353, params['p122'])

    # 354. fused_reshape_add_reshape_transpose_reshape_transpose_17
    # kernel 0
    output_size = 1572864
    nodes[354] = kaas.bufferSpec('354', output_size, const=False, ephemeral=True)
    arguments = [(nodes[354], 'o'), (nodes[352], 'i'), (nodes[353], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 355. fused_nn_batch_matmul_416
    # kernel 0
    output_size = 1572864
    nodes[355] = kaas.bufferSpec('355', output_size, const=False, ephemeral=True)
    arguments = [(nodes[350], 'i'), (nodes[354], 'i'), (nodes[355], 'o'), ]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 356. fused_reshape_transpose_reshape16
    # kernel 0
    output_size = 1572864
    nodes[356] = kaas.bufferSpec('356', output_size, const=False, ephemeral=True)
    arguments = [(nodes[356], 'o'), (nodes[355], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 357. p123
    nodes[357] = addToKV(357, params['p123'])

    # 358. fused_nn_batch_matmul_332
    # kernel 0
    output_size = 1572864
    nodes[358] = kaas.bufferSpec('358', output_size, const=False, ephemeral=True)
    arguments = [(nodes[356], 'i'), (nodes[357], 'i'), (nodes[358], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 359. p124
    nodes[359] = addToKV(359, params['p124'])

    # 360. fused_reshape_add_add33
    # kernel 0
    output_size = 1572864
    nodes[360] = kaas.bufferSpec('360', output_size, const=False, ephemeral=True)
    arguments = [(nodes[360], 'o'), (nodes[358], 'i'), (nodes[359], 'i'), (nodes[335], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 361. fused_mean15
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a30', output_size, const=False, ephemeral=True))
    arguments = [(nodes[360], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[361] = kaas.bufferSpec('361', output_size, const=False, ephemeral=True)
    arguments = [(nodes[361], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 362. fused_subtract33
    # kernel 0
    output_size = 1572864
    nodes[362] = kaas.bufferSpec('362', output_size, const=False, ephemeral=True)
    arguments = [(nodes[362], 'o'), (nodes[360], 'i'), (nodes[361], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 363. fused_power_mean33
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a31', output_size, const=False, ephemeral=True))
    arguments = [(nodes[362], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[363] = kaas.bufferSpec('363', output_size, const=False, ephemeral=True)
    arguments = [(nodes[363], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 364. p125
    nodes[364] = addToKV(364, params['p125'])

    # 365. p126
    nodes[365] = addToKV(365, params['p126'])

    # 366. fused_add_sqrt_divide_multiply_add32
    # kernel 0
    output_size = 1572864
    nodes[366] = kaas.bufferSpec('366', output_size, const=False, ephemeral=True)
    arguments = [(nodes[366], 'o'), (nodes[363], 'i'), (nodes[362], 'i'), (nodes[364], 'i'), (nodes[365], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 367. reshape_nop
    nodes[367] = nodes[366]

    # 368. p127
    nodes[368] = addToKV(368, params['p127'])

    # 369. fused_nn_batch_matmul_216
    # kernel 0
    output_size = 6291456
    nodes[369] = kaas.bufferSpec('369', output_size, const=False, ephemeral=True)
    arguments = [(nodes[367], 'i'), (nodes[368], 'i'), (nodes[369], 'o'), ]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 370. p128
    nodes[370] = addToKV(370, params['p128'])

    # 371. fused_reshape_add_multiply_divide_erf_add_multiply_reshape16
    # kernel 0
    output_size = 6291456
    nodes[371] = kaas.bufferSpec('371', output_size, const=False, ephemeral=True)
    arguments = [(nodes[371], 'o'), (nodes[369], 'i'), (nodes[370], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 372. p129
    nodes[372] = addToKV(372, params['p129'])

    # 373. fused_nn_batch_matmul_116
    # kernel 0
    output_size = 1572864
    nodes[373] = kaas.bufferSpec('373', output_size, const=False, ephemeral=True)
    arguments = [(nodes[371], 'i'), (nodes[372], 'i'), (nodes[373], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 374. p130
    nodes[374] = addToKV(374, params['p130'])

    # 375. fused_reshape_add_add32
    # kernel 0
    output_size = 1572864
    nodes[375] = kaas.bufferSpec('375', output_size, const=False, ephemeral=True)
    arguments = [(nodes[375], 'o'), (nodes[373], 'i'), (nodes[374], 'i'), (nodes[366], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 376. fused_mean16
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a32', output_size, const=False, ephemeral=True))
    arguments = [(nodes[375], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[376] = kaas.bufferSpec('376', output_size, const=False, ephemeral=True)
    arguments = [(nodes[376], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 377. fused_subtract32
    # kernel 0
    output_size = 1572864
    nodes[377] = kaas.bufferSpec('377', output_size, const=False, ephemeral=True)
    arguments = [(nodes[377], 'o'), (nodes[375], 'i'), (nodes[376], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 378. fused_power_mean32
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a33', output_size, const=False, ephemeral=True))
    arguments = [(nodes[377], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[378] = kaas.bufferSpec('378', output_size, const=False, ephemeral=True)
    arguments = [(nodes[378], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 379. p131
    nodes[379] = addToKV(379, params['p131'])

    # 380. p132
    nodes[380] = addToKV(380, params['p132'])

    # 381. fused_add_sqrt_divide_multiply_add31
    # kernel 0
    output_size = 1572864
    nodes[381] = kaas.bufferSpec('381', output_size, const=False, ephemeral=True)
    arguments = [(nodes[381], 'o'), (nodes[378], 'i'), (nodes[377], 'i'), (nodes[379], 'i'), (nodes[380], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 382. reshape_nop
    nodes[382] = nodes[381]

    # 383. p133
    nodes[383] = addToKV(383, params['p133'])

    # 384. fused_nn_batch_matmul_331
    # kernel 0
    output_size = 1572864
    nodes[384] = kaas.bufferSpec('384', output_size, const=False, ephemeral=True)
    arguments = [(nodes[382], 'i'), (nodes[383], 'i'), (nodes[384], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 385. p134
    nodes[385] = addToKV(385, params['p134'])

    # 386. fused_reshape_add_reshape_transpose_reshape15
    # kernel 0
    output_size = 1572864
    nodes[386] = kaas.bufferSpec('386', output_size, const=False, ephemeral=True)
    arguments = [(nodes[386], 'o'), (nodes[384], 'i'), (nodes[385], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 387. p135
    nodes[387] = addToKV(387, params['p135'])

    # 388. fused_nn_batch_matmul_364
    # kernel 0
    output_size = 1572864
    nodes[388] = kaas.bufferSpec('388', output_size, const=False, ephemeral=True)
    arguments = [(nodes[382], 'i'), (nodes[387], 'i'), (nodes[388], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 389. p136
    nodes[389] = addToKV(389, params['p136'])

    # 390. fused_reshape_add_reshape_transpose_reshape_transpose8
    # kernel 0
    output_size = 1572864
    nodes[390] = kaas.bufferSpec('390', output_size, const=False, ephemeral=True)
    arguments = [(nodes[390], 'o'), (nodes[388], 'i'), (nodes[389], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 391. fused_nn_batch_matmul_515
    # kernel 0
    output_size = 9437184
    nodes[391] = kaas.bufferSpec('391', output_size, const=False, ephemeral=True)
    arguments = [(nodes[386], 'i'), (nodes[390], 'i'), (nodes[391], 'o'), ]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 392. fused_reshape_divide_add15
    # kernel 0
    output_size = 9437184
    nodes[392] = kaas.bufferSpec('392', output_size, const=False, ephemeral=True)
    arguments = [(nodes[392], 'o'), (nodes[391], 'i'), (nodes[23], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 393. fused_max8
    # kernel 0
    output_size = 24576
    nodes[393] = kaas.bufferSpec('393', output_size, const=False, ephemeral=True)
    arguments = [(nodes[392], 'i'), (nodes[393], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 394. fused_subtract_exp15
    # kernel 0
    output_size = 9437184
    nodes[394] = kaas.bufferSpec('394', output_size, const=False, ephemeral=True)
    arguments = [(nodes[394], 'o'), (nodes[392], 'i'), (nodes[393], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 395. fused_sum8
    # kernel 0
    output_size = 24576
    nodes[395] = kaas.bufferSpec('395', output_size, const=False, ephemeral=True)
    arguments = [(nodes[394], 'i'), (nodes[395], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 396. fused_divide_reshape15
    # kernel 0
    output_size = 9437184
    nodes[396] = kaas.bufferSpec('396', output_size, const=False, ephemeral=True)
    arguments = [(nodes[396], 'o'), (nodes[394], 'i'), (nodes[395], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 397. p137
    nodes[397] = addToKV(397, params['p137'])

    # 398. fused_nn_batch_matmul_365
    # kernel 0
    output_size = 1572864
    nodes[398] = kaas.bufferSpec('398', output_size, const=False, ephemeral=True)
    arguments = [(nodes[382], 'i'), (nodes[397], 'i'), (nodes[398], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 399. p138
    nodes[399] = addToKV(399, params['p138'])

    # 400. fused_reshape_add_reshape_transpose_reshape_transpose_18
    # kernel 0
    output_size = 1572864
    nodes[400] = kaas.bufferSpec('400', output_size, const=False, ephemeral=True)
    arguments = [(nodes[400], 'o'), (nodes[398], 'i'), (nodes[399], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 401. fused_nn_batch_matmul_415
    # kernel 0
    output_size = 1572864
    nodes[401] = kaas.bufferSpec('401', output_size, const=False, ephemeral=True)
    arguments = [(nodes[396], 'i'), (nodes[400], 'i'), (nodes[401], 'o'), ]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 402. fused_reshape_transpose_reshape15
    # kernel 0
    output_size = 1572864
    nodes[402] = kaas.bufferSpec('402', output_size, const=False, ephemeral=True)
    arguments = [(nodes[402], 'o'), (nodes[401], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 403. p139
    nodes[403] = addToKV(403, params['p139'])

    # 404. fused_nn_batch_matmul_330
    # kernel 0
    output_size = 1572864
    nodes[404] = kaas.bufferSpec('404', output_size, const=False, ephemeral=True)
    arguments = [(nodes[402], 'i'), (nodes[403], 'i'), (nodes[404], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 405. p140
    nodes[405] = addToKV(405, params['p140'])

    # 406. fused_reshape_add_add31
    # kernel 0
    output_size = 1572864
    nodes[406] = kaas.bufferSpec('406', output_size, const=False, ephemeral=True)
    arguments = [(nodes[406], 'o'), (nodes[404], 'i'), (nodes[405], 'i'), (nodes[381], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 407. fused_mean17
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a34', output_size, const=False, ephemeral=True))
    arguments = [(nodes[406], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[407] = kaas.bufferSpec('407', output_size, const=False, ephemeral=True)
    arguments = [(nodes[407], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 408. fused_subtract31
    # kernel 0
    output_size = 1572864
    nodes[408] = kaas.bufferSpec('408', output_size, const=False, ephemeral=True)
    arguments = [(nodes[408], 'o'), (nodes[406], 'i'), (nodes[407], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 409. fused_power_mean31
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a35', output_size, const=False, ephemeral=True))
    arguments = [(nodes[408], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[409] = kaas.bufferSpec('409', output_size, const=False, ephemeral=True)
    arguments = [(nodes[409], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 410. p141
    nodes[410] = addToKV(410, params['p141'])

    # 411. p142
    nodes[411] = addToKV(411, params['p142'])

    # 412. fused_add_sqrt_divide_multiply_add30
    # kernel 0
    output_size = 1572864
    nodes[412] = kaas.bufferSpec('412', output_size, const=False, ephemeral=True)
    arguments = [(nodes[412], 'o'), (nodes[409], 'i'), (nodes[408], 'i'), (nodes[410], 'i'), (nodes[411], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 413. reshape_nop
    nodes[413] = nodes[412]

    # 414. p143
    nodes[414] = addToKV(414, params['p143'])

    # 415. fused_nn_batch_matmul_215
    # kernel 0
    output_size = 6291456
    nodes[415] = kaas.bufferSpec('415', output_size, const=False, ephemeral=True)
    arguments = [(nodes[413], 'i'), (nodes[414], 'i'), (nodes[415], 'o'), ]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 416. p144
    nodes[416] = addToKV(416, params['p144'])

    # 417. fused_reshape_add_multiply_divide_erf_add_multiply_reshape15
    # kernel 0
    output_size = 6291456
    nodes[417] = kaas.bufferSpec('417', output_size, const=False, ephemeral=True)
    arguments = [(nodes[417], 'o'), (nodes[415], 'i'), (nodes[416], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 418. p145
    nodes[418] = addToKV(418, params['p145'])

    # 419. fused_nn_batch_matmul_115
    # kernel 0
    output_size = 1572864
    nodes[419] = kaas.bufferSpec('419', output_size, const=False, ephemeral=True)
    arguments = [(nodes[417], 'i'), (nodes[418], 'i'), (nodes[419], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 420. p146
    nodes[420] = addToKV(420, params['p146'])

    # 421. fused_reshape_add_add30
    # kernel 0
    output_size = 1572864
    nodes[421] = kaas.bufferSpec('421', output_size, const=False, ephemeral=True)
    arguments = [(nodes[421], 'o'), (nodes[419], 'i'), (nodes[420], 'i'), (nodes[412], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 422. fused_mean18
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a36', output_size, const=False, ephemeral=True))
    arguments = [(nodes[421], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[422] = kaas.bufferSpec('422', output_size, const=False, ephemeral=True)
    arguments = [(nodes[422], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 423. fused_subtract30
    # kernel 0
    output_size = 1572864
    nodes[423] = kaas.bufferSpec('423', output_size, const=False, ephemeral=True)
    arguments = [(nodes[423], 'o'), (nodes[421], 'i'), (nodes[422], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 424. fused_power_mean30
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a37', output_size, const=False, ephemeral=True))
    arguments = [(nodes[423], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[424] = kaas.bufferSpec('424', output_size, const=False, ephemeral=True)
    arguments = [(nodes[424], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 425. p147
    nodes[425] = addToKV(425, params['p147'])

    # 426. p148
    nodes[426] = addToKV(426, params['p148'])

    # 427. fused_add_sqrt_divide_multiply_add29
    # kernel 0
    output_size = 1572864
    nodes[427] = kaas.bufferSpec('427', output_size, const=False, ephemeral=True)
    arguments = [(nodes[427], 'o'), (nodes[424], 'i'), (nodes[423], 'i'), (nodes[425], 'i'), (nodes[426], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 428. reshape_nop
    nodes[428] = nodes[427]

    # 429. p149
    nodes[429] = addToKV(429, params['p149'])

    # 430. fused_nn_batch_matmul_329
    # kernel 0
    output_size = 1572864
    nodes[430] = kaas.bufferSpec('430', output_size, const=False, ephemeral=True)
    arguments = [(nodes[428], 'i'), (nodes[429], 'i'), (nodes[430], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 431. p150
    nodes[431] = addToKV(431, params['p150'])

    # 432. fused_reshape_add_reshape_transpose_reshape14
    # kernel 0
    output_size = 1572864
    nodes[432] = kaas.bufferSpec('432', output_size, const=False, ephemeral=True)
    arguments = [(nodes[432], 'o'), (nodes[430], 'i'), (nodes[431], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 433. p151
    nodes[433] = addToKV(433, params['p151'])

    # 434. fused_nn_batch_matmul_366
    # kernel 0
    output_size = 1572864
    nodes[434] = kaas.bufferSpec('434', output_size, const=False, ephemeral=True)
    arguments = [(nodes[428], 'i'), (nodes[433], 'i'), (nodes[434], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 435. p152
    nodes[435] = addToKV(435, params['p152'])

    # 436. fused_reshape_add_reshape_transpose_reshape_transpose9
    # kernel 0
    output_size = 1572864
    nodes[436] = kaas.bufferSpec('436', output_size, const=False, ephemeral=True)
    arguments = [(nodes[436], 'o'), (nodes[434], 'i'), (nodes[435], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 437. fused_nn_batch_matmul_514
    # kernel 0
    output_size = 9437184
    nodes[437] = kaas.bufferSpec('437', output_size, const=False, ephemeral=True)
    arguments = [(nodes[432], 'i'), (nodes[436], 'i'), (nodes[437], 'o'), ]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 438. fused_reshape_divide_add14
    # kernel 0
    output_size = 9437184
    nodes[438] = kaas.bufferSpec('438', output_size, const=False, ephemeral=True)
    arguments = [(nodes[438], 'o'), (nodes[437], 'i'), (nodes[23], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 439. fused_max9
    # kernel 0
    output_size = 24576
    nodes[439] = kaas.bufferSpec('439', output_size, const=False, ephemeral=True)
    arguments = [(nodes[438], 'i'), (nodes[439], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 440. fused_subtract_exp14
    # kernel 0
    output_size = 9437184
    nodes[440] = kaas.bufferSpec('440', output_size, const=False, ephemeral=True)
    arguments = [(nodes[440], 'o'), (nodes[438], 'i'), (nodes[439], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 441. fused_sum9
    # kernel 0
    output_size = 24576
    nodes[441] = kaas.bufferSpec('441', output_size, const=False, ephemeral=True)
    arguments = [(nodes[440], 'i'), (nodes[441], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 442. fused_divide_reshape14
    # kernel 0
    output_size = 9437184
    nodes[442] = kaas.bufferSpec('442', output_size, const=False, ephemeral=True)
    arguments = [(nodes[442], 'o'), (nodes[440], 'i'), (nodes[441], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 443. p153
    nodes[443] = addToKV(443, params['p153'])

    # 444. fused_nn_batch_matmul_367
    # kernel 0
    output_size = 1572864
    nodes[444] = kaas.bufferSpec('444', output_size, const=False, ephemeral=True)
    arguments = [(nodes[428], 'i'), (nodes[443], 'i'), (nodes[444], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 445. p154
    nodes[445] = addToKV(445, params['p154'])

    # 446. fused_reshape_add_reshape_transpose_reshape_transpose_19
    # kernel 0
    output_size = 1572864
    nodes[446] = kaas.bufferSpec('446', output_size, const=False, ephemeral=True)
    arguments = [(nodes[446], 'o'), (nodes[444], 'i'), (nodes[445], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 447. fused_nn_batch_matmul_414
    # kernel 0
    output_size = 1572864
    nodes[447] = kaas.bufferSpec('447', output_size, const=False, ephemeral=True)
    arguments = [(nodes[442], 'i'), (nodes[446], 'i'), (nodes[447], 'o'), ]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 448. fused_reshape_transpose_reshape14
    # kernel 0
    output_size = 1572864
    nodes[448] = kaas.bufferSpec('448', output_size, const=False, ephemeral=True)
    arguments = [(nodes[448], 'o'), (nodes[447], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 449. p155
    nodes[449] = addToKV(449, params['p155'])

    # 450. fused_nn_batch_matmul_328
    # kernel 0
    output_size = 1572864
    nodes[450] = kaas.bufferSpec('450', output_size, const=False, ephemeral=True)
    arguments = [(nodes[448], 'i'), (nodes[449], 'i'), (nodes[450], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 451. p156
    nodes[451] = addToKV(451, params['p156'])

    # 452. fused_reshape_add_add29
    # kernel 0
    output_size = 1572864
    nodes[452] = kaas.bufferSpec('452', output_size, const=False, ephemeral=True)
    arguments = [(nodes[452], 'o'), (nodes[450], 'i'), (nodes[451], 'i'), (nodes[427], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 453. fused_mean19
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a38', output_size, const=False, ephemeral=True))
    arguments = [(nodes[452], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[453] = kaas.bufferSpec('453', output_size, const=False, ephemeral=True)
    arguments = [(nodes[453], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 454. fused_subtract29
    # kernel 0
    output_size = 1572864
    nodes[454] = kaas.bufferSpec('454', output_size, const=False, ephemeral=True)
    arguments = [(nodes[454], 'o'), (nodes[452], 'i'), (nodes[453], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 455. fused_power_mean29
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a39', output_size, const=False, ephemeral=True))
    arguments = [(nodes[454], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[455] = kaas.bufferSpec('455', output_size, const=False, ephemeral=True)
    arguments = [(nodes[455], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 456. p157
    nodes[456] = addToKV(456, params['p157'])

    # 457. p158
    nodes[457] = addToKV(457, params['p158'])

    # 458. fused_add_sqrt_divide_multiply_add28
    # kernel 0
    output_size = 1572864
    nodes[458] = kaas.bufferSpec('458', output_size, const=False, ephemeral=True)
    arguments = [(nodes[458], 'o'), (nodes[455], 'i'), (nodes[454], 'i'), (nodes[456], 'i'), (nodes[457], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 459. reshape_nop
    nodes[459] = nodes[458]

    # 460. p159
    nodes[460] = addToKV(460, params['p159'])

    # 461. fused_nn_batch_matmul_214
    # kernel 0
    output_size = 6291456
    nodes[461] = kaas.bufferSpec('461', output_size, const=False, ephemeral=True)
    arguments = [(nodes[459], 'i'), (nodes[460], 'i'), (nodes[461], 'o'), ]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 462. p160
    nodes[462] = addToKV(462, params['p160'])

    # 463. fused_reshape_add_multiply_divide_erf_add_multiply_reshape14
    # kernel 0
    output_size = 6291456
    nodes[463] = kaas.bufferSpec('463', output_size, const=False, ephemeral=True)
    arguments = [(nodes[463], 'o'), (nodes[461], 'i'), (nodes[462], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 464. p161
    nodes[464] = addToKV(464, params['p161'])

    # 465. fused_nn_batch_matmul_114
    # kernel 0
    output_size = 1572864
    nodes[465] = kaas.bufferSpec('465', output_size, const=False, ephemeral=True)
    arguments = [(nodes[463], 'i'), (nodes[464], 'i'), (nodes[465], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 466. p162
    nodes[466] = addToKV(466, params['p162'])

    # 467. fused_reshape_add_add28
    # kernel 0
    output_size = 1572864
    nodes[467] = kaas.bufferSpec('467', output_size, const=False, ephemeral=True)
    arguments = [(nodes[467], 'o'), (nodes[465], 'i'), (nodes[466], 'i'), (nodes[458], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 468. fused_mean20
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a40', output_size, const=False, ephemeral=True))
    arguments = [(nodes[467], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[468] = kaas.bufferSpec('468', output_size, const=False, ephemeral=True)
    arguments = [(nodes[468], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 469. fused_subtract28
    # kernel 0
    output_size = 1572864
    nodes[469] = kaas.bufferSpec('469', output_size, const=False, ephemeral=True)
    arguments = [(nodes[469], 'o'), (nodes[467], 'i'), (nodes[468], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 470. fused_power_mean28
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a41', output_size, const=False, ephemeral=True))
    arguments = [(nodes[469], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[470] = kaas.bufferSpec('470', output_size, const=False, ephemeral=True)
    arguments = [(nodes[470], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 471. p163
    nodes[471] = addToKV(471, params['p163'])

    # 472. p164
    nodes[472] = addToKV(472, params['p164'])

    # 473. fused_add_sqrt_divide_multiply_add27
    # kernel 0
    output_size = 1572864
    nodes[473] = kaas.bufferSpec('473', output_size, const=False, ephemeral=True)
    arguments = [(nodes[473], 'o'), (nodes[470], 'i'), (nodes[469], 'i'), (nodes[471], 'i'), (nodes[472], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 474. reshape_nop
    nodes[474] = nodes[473]

    # 475. p165
    nodes[475] = addToKV(475, params['p165'])

    # 476. fused_nn_batch_matmul_327
    # kernel 0
    output_size = 1572864
    nodes[476] = kaas.bufferSpec('476', output_size, const=False, ephemeral=True)
    arguments = [(nodes[474], 'i'), (nodes[475], 'i'), (nodes[476], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 477. p166
    nodes[477] = addToKV(477, params['p166'])

    # 478. fused_reshape_add_reshape_transpose_reshape13
    # kernel 0
    output_size = 1572864
    nodes[478] = kaas.bufferSpec('478', output_size, const=False, ephemeral=True)
    arguments = [(nodes[478], 'o'), (nodes[476], 'i'), (nodes[477], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 479. p167
    nodes[479] = addToKV(479, params['p167'])

    # 480. fused_nn_batch_matmul_368
    # kernel 0
    output_size = 1572864
    nodes[480] = kaas.bufferSpec('480', output_size, const=False, ephemeral=True)
    arguments = [(nodes[474], 'i'), (nodes[479], 'i'), (nodes[480], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 481. p168
    nodes[481] = addToKV(481, params['p168'])

    # 482. fused_reshape_add_reshape_transpose_reshape_transpose10
    # kernel 0
    output_size = 1572864
    nodes[482] = kaas.bufferSpec('482', output_size, const=False, ephemeral=True)
    arguments = [(nodes[482], 'o'), (nodes[480], 'i'), (nodes[481], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 483. fused_nn_batch_matmul_513
    # kernel 0
    output_size = 9437184
    nodes[483] = kaas.bufferSpec('483', output_size, const=False, ephemeral=True)
    arguments = [(nodes[478], 'i'), (nodes[482], 'i'), (nodes[483], 'o'), ]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 484. fused_reshape_divide_add13
    # kernel 0
    output_size = 9437184
    nodes[484] = kaas.bufferSpec('484', output_size, const=False, ephemeral=True)
    arguments = [(nodes[484], 'o'), (nodes[483], 'i'), (nodes[23], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 485. fused_max10
    # kernel 0
    output_size = 24576
    nodes[485] = kaas.bufferSpec('485', output_size, const=False, ephemeral=True)
    arguments = [(nodes[484], 'i'), (nodes[485], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 486. fused_subtract_exp13
    # kernel 0
    output_size = 9437184
    nodes[486] = kaas.bufferSpec('486', output_size, const=False, ephemeral=True)
    arguments = [(nodes[486], 'o'), (nodes[484], 'i'), (nodes[485], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 487. fused_sum10
    # kernel 0
    output_size = 24576
    nodes[487] = kaas.bufferSpec('487', output_size, const=False, ephemeral=True)
    arguments = [(nodes[486], 'i'), (nodes[487], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 488. fused_divide_reshape13
    # kernel 0
    output_size = 9437184
    nodes[488] = kaas.bufferSpec('488', output_size, const=False, ephemeral=True)
    arguments = [(nodes[488], 'o'), (nodes[486], 'i'), (nodes[487], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 489. p169
    nodes[489] = addToKV(489, params['p169'])

    # 490. fused_nn_batch_matmul_369
    # kernel 0
    output_size = 1572864
    nodes[490] = kaas.bufferSpec('490', output_size, const=False, ephemeral=True)
    arguments = [(nodes[474], 'i'), (nodes[489], 'i'), (nodes[490], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 491. p170
    nodes[491] = addToKV(491, params['p170'])

    # 492. fused_reshape_add_reshape_transpose_reshape_transpose_110
    # kernel 0
    output_size = 1572864
    nodes[492] = kaas.bufferSpec('492', output_size, const=False, ephemeral=True)
    arguments = [(nodes[492], 'o'), (nodes[490], 'i'), (nodes[491], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 493. fused_nn_batch_matmul_413
    # kernel 0
    output_size = 1572864
    nodes[493] = kaas.bufferSpec('493', output_size, const=False, ephemeral=True)
    arguments = [(nodes[488], 'i'), (nodes[492], 'i'), (nodes[493], 'o'), ]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 494. fused_reshape_transpose_reshape13
    # kernel 0
    output_size = 1572864
    nodes[494] = kaas.bufferSpec('494', output_size, const=False, ephemeral=True)
    arguments = [(nodes[494], 'o'), (nodes[493], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 495. p171
    nodes[495] = addToKV(495, params['p171'])

    # 496. fused_nn_batch_matmul_326
    # kernel 0
    output_size = 1572864
    nodes[496] = kaas.bufferSpec('496', output_size, const=False, ephemeral=True)
    arguments = [(nodes[494], 'i'), (nodes[495], 'i'), (nodes[496], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 497. p172
    nodes[497] = addToKV(497, params['p172'])

    # 498. fused_reshape_add_add27
    # kernel 0
    output_size = 1572864
    nodes[498] = kaas.bufferSpec('498', output_size, const=False, ephemeral=True)
    arguments = [(nodes[498], 'o'), (nodes[496], 'i'), (nodes[497], 'i'), (nodes[473], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 499. fused_mean21
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a42', output_size, const=False, ephemeral=True))
    arguments = [(nodes[498], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[499] = kaas.bufferSpec('499', output_size, const=False, ephemeral=True)
    arguments = [(nodes[499], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 500. fused_subtract27
    # kernel 0
    output_size = 1572864
    nodes[500] = kaas.bufferSpec('500', output_size, const=False, ephemeral=True)
    arguments = [(nodes[500], 'o'), (nodes[498], 'i'), (nodes[499], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 501. fused_power_mean27
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a43', output_size, const=False, ephemeral=True))
    arguments = [(nodes[500], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[501] = kaas.bufferSpec('501', output_size, const=False, ephemeral=True)
    arguments = [(nodes[501], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 502. p173
    nodes[502] = addToKV(502, params['p173'])

    # 503. p174
    nodes[503] = addToKV(503, params['p174'])

    # 504. fused_add_sqrt_divide_multiply_add26
    # kernel 0
    output_size = 1572864
    nodes[504] = kaas.bufferSpec('504', output_size, const=False, ephemeral=True)
    arguments = [(nodes[504], 'o'), (nodes[501], 'i'), (nodes[500], 'i'), (nodes[502], 'i'), (nodes[503], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 505. reshape_nop
    nodes[505] = nodes[504]

    # 506. p175
    nodes[506] = addToKV(506, params['p175'])

    # 507. fused_nn_batch_matmul_213
    # kernel 0
    output_size = 6291456
    nodes[507] = kaas.bufferSpec('507', output_size, const=False, ephemeral=True)
    arguments = [(nodes[505], 'i'), (nodes[506], 'i'), (nodes[507], 'o'), ]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 508. p176
    nodes[508] = addToKV(508, params['p176'])

    # 509. fused_reshape_add_multiply_divide_erf_add_multiply_reshape13
    # kernel 0
    output_size = 6291456
    nodes[509] = kaas.bufferSpec('509', output_size, const=False, ephemeral=True)
    arguments = [(nodes[509], 'o'), (nodes[507], 'i'), (nodes[508], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 510. p177
    nodes[510] = addToKV(510, params['p177'])

    # 511. fused_nn_batch_matmul_113
    # kernel 0
    output_size = 1572864
    nodes[511] = kaas.bufferSpec('511', output_size, const=False, ephemeral=True)
    arguments = [(nodes[509], 'i'), (nodes[510], 'i'), (nodes[511], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 512. p178
    nodes[512] = addToKV(512, params['p178'])

    # 513. fused_reshape_add_add26
    # kernel 0
    output_size = 1572864
    nodes[513] = kaas.bufferSpec('513', output_size, const=False, ephemeral=True)
    arguments = [(nodes[513], 'o'), (nodes[511], 'i'), (nodes[512], 'i'), (nodes[504], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 514. fused_mean22
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a44', output_size, const=False, ephemeral=True))
    arguments = [(nodes[513], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[514] = kaas.bufferSpec('514', output_size, const=False, ephemeral=True)
    arguments = [(nodes[514], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 515. fused_subtract26
    # kernel 0
    output_size = 1572864
    nodes[515] = kaas.bufferSpec('515', output_size, const=False, ephemeral=True)
    arguments = [(nodes[515], 'o'), (nodes[513], 'i'), (nodes[514], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 516. fused_power_mean26
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a45', output_size, const=False, ephemeral=True))
    arguments = [(nodes[515], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[516] = kaas.bufferSpec('516', output_size, const=False, ephemeral=True)
    arguments = [(nodes[516], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 517. p179
    nodes[517] = addToKV(517, params['p179'])

    # 518. p180
    nodes[518] = addToKV(518, params['p180'])

    # 519. fused_add_sqrt_divide_multiply_add25
    # kernel 0
    output_size = 1572864
    nodes[519] = kaas.bufferSpec('519', output_size, const=False, ephemeral=True)
    arguments = [(nodes[519], 'o'), (nodes[516], 'i'), (nodes[515], 'i'), (nodes[517], 'i'), (nodes[518], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 520. reshape_nop
    nodes[520] = nodes[519]

    # 521. p181
    nodes[521] = addToKV(521, params['p181'])

    # 522. fused_nn_batch_matmul_325
    # kernel 0
    output_size = 1572864
    nodes[522] = kaas.bufferSpec('522', output_size, const=False, ephemeral=True)
    arguments = [(nodes[520], 'i'), (nodes[521], 'i'), (nodes[522], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 523. p182
    nodes[523] = addToKV(523, params['p182'])

    # 524. fused_reshape_add_reshape_transpose_reshape12
    # kernel 0
    output_size = 1572864
    nodes[524] = kaas.bufferSpec('524', output_size, const=False, ephemeral=True)
    arguments = [(nodes[524], 'o'), (nodes[522], 'i'), (nodes[523], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 525. p183
    nodes[525] = addToKV(525, params['p183'])

    # 526. fused_nn_batch_matmul_370
    # kernel 0
    output_size = 1572864
    nodes[526] = kaas.bufferSpec('526', output_size, const=False, ephemeral=True)
    arguments = [(nodes[520], 'i'), (nodes[525], 'i'), (nodes[526], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 527. p184
    nodes[527] = addToKV(527, params['p184'])

    # 528. fused_reshape_add_reshape_transpose_reshape_transpose11
    # kernel 0
    output_size = 1572864
    nodes[528] = kaas.bufferSpec('528', output_size, const=False, ephemeral=True)
    arguments = [(nodes[528], 'o'), (nodes[526], 'i'), (nodes[527], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 529. fused_nn_batch_matmul_512
    # kernel 0
    output_size = 9437184
    nodes[529] = kaas.bufferSpec('529', output_size, const=False, ephemeral=True)
    arguments = [(nodes[524], 'i'), (nodes[528], 'i'), (nodes[529], 'o'), ]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 530. fused_reshape_divide_add12
    # kernel 0
    output_size = 9437184
    nodes[530] = kaas.bufferSpec('530', output_size, const=False, ephemeral=True)
    arguments = [(nodes[530], 'o'), (nodes[529], 'i'), (nodes[23], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 531. fused_max11
    # kernel 0
    output_size = 24576
    nodes[531] = kaas.bufferSpec('531', output_size, const=False, ephemeral=True)
    arguments = [(nodes[530], 'i'), (nodes[531], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 532. fused_subtract_exp12
    # kernel 0
    output_size = 9437184
    nodes[532] = kaas.bufferSpec('532', output_size, const=False, ephemeral=True)
    arguments = [(nodes[532], 'o'), (nodes[530], 'i'), (nodes[531], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 533. fused_sum11
    # kernel 0
    output_size = 24576
    nodes[533] = kaas.bufferSpec('533', output_size, const=False, ephemeral=True)
    arguments = [(nodes[532], 'i'), (nodes[533], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 534. fused_divide_reshape12
    # kernel 0
    output_size = 9437184
    nodes[534] = kaas.bufferSpec('534', output_size, const=False, ephemeral=True)
    arguments = [(nodes[534], 'o'), (nodes[532], 'i'), (nodes[533], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 535. p185
    nodes[535] = addToKV(535, params['p185'])

    # 536. fused_nn_batch_matmul_371
    # kernel 0
    output_size = 1572864
    nodes[536] = kaas.bufferSpec('536', output_size, const=False, ephemeral=True)
    arguments = [(nodes[520], 'i'), (nodes[535], 'i'), (nodes[536], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 537. p186
    nodes[537] = addToKV(537, params['p186'])

    # 538. fused_reshape_add_reshape_transpose_reshape_transpose_111
    # kernel 0
    output_size = 1572864
    nodes[538] = kaas.bufferSpec('538', output_size, const=False, ephemeral=True)
    arguments = [(nodes[538], 'o'), (nodes[536], 'i'), (nodes[537], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 539. fused_nn_batch_matmul_412
    # kernel 0
    output_size = 1572864
    nodes[539] = kaas.bufferSpec('539', output_size, const=False, ephemeral=True)
    arguments = [(nodes[534], 'i'), (nodes[538], 'i'), (nodes[539], 'o'), ]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 540. fused_reshape_transpose_reshape12
    # kernel 0
    output_size = 1572864
    nodes[540] = kaas.bufferSpec('540', output_size, const=False, ephemeral=True)
    arguments = [(nodes[540], 'o'), (nodes[539], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 541. p187
    nodes[541] = addToKV(541, params['p187'])

    # 542. fused_nn_batch_matmul_324
    # kernel 0
    output_size = 1572864
    nodes[542] = kaas.bufferSpec('542', output_size, const=False, ephemeral=True)
    arguments = [(nodes[540], 'i'), (nodes[541], 'i'), (nodes[542], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 543. p188
    nodes[543] = addToKV(543, params['p188'])

    # 544. fused_reshape_add_add25
    # kernel 0
    output_size = 1572864
    nodes[544] = kaas.bufferSpec('544', output_size, const=False, ephemeral=True)
    arguments = [(nodes[544], 'o'), (nodes[542], 'i'), (nodes[543], 'i'), (nodes[519], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 545. fused_mean23
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a46', output_size, const=False, ephemeral=True))
    arguments = [(nodes[544], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[545] = kaas.bufferSpec('545', output_size, const=False, ephemeral=True)
    arguments = [(nodes[545], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 546. fused_subtract25
    # kernel 0
    output_size = 1572864
    nodes[546] = kaas.bufferSpec('546', output_size, const=False, ephemeral=True)
    arguments = [(nodes[546], 'o'), (nodes[544], 'i'), (nodes[545], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 547. fused_power_mean25
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a47', output_size, const=False, ephemeral=True))
    arguments = [(nodes[546], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[547] = kaas.bufferSpec('547', output_size, const=False, ephemeral=True)
    arguments = [(nodes[547], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 548. p189
    nodes[548] = addToKV(548, params['p189'])

    # 549. p190
    nodes[549] = addToKV(549, params['p190'])

    # 550. fused_add_sqrt_divide_multiply_add24
    # kernel 0
    output_size = 1572864
    nodes[550] = kaas.bufferSpec('550', output_size, const=False, ephemeral=True)
    arguments = [(nodes[550], 'o'), (nodes[547], 'i'), (nodes[546], 'i'), (nodes[548], 'i'), (nodes[549], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 551. reshape_nop
    nodes[551] = nodes[550]

    # 552. p191
    nodes[552] = addToKV(552, params['p191'])

    # 553. fused_nn_batch_matmul_212
    # kernel 0
    output_size = 6291456
    nodes[553] = kaas.bufferSpec('553', output_size, const=False, ephemeral=True)
    arguments = [(nodes[551], 'i'), (nodes[552], 'i'), (nodes[553], 'o'), ]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 554. p192
    nodes[554] = addToKV(554, params['p192'])

    # 555. fused_reshape_add_multiply_divide_erf_add_multiply_reshape12
    # kernel 0
    output_size = 6291456
    nodes[555] = kaas.bufferSpec('555', output_size, const=False, ephemeral=True)
    arguments = [(nodes[555], 'o'), (nodes[553], 'i'), (nodes[554], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 556. p193
    nodes[556] = addToKV(556, params['p193'])

    # 557. fused_nn_batch_matmul_112
    # kernel 0
    output_size = 1572864
    nodes[557] = kaas.bufferSpec('557', output_size, const=False, ephemeral=True)
    arguments = [(nodes[555], 'i'), (nodes[556], 'i'), (nodes[557], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 558. p194
    nodes[558] = addToKV(558, params['p194'])

    # 559. fused_reshape_add_add24
    # kernel 0
    output_size = 1572864
    nodes[559] = kaas.bufferSpec('559', output_size, const=False, ephemeral=True)
    arguments = [(nodes[559], 'o'), (nodes[557], 'i'), (nodes[558], 'i'), (nodes[550], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 560. fused_mean24
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a48', output_size, const=False, ephemeral=True))
    arguments = [(nodes[559], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[560] = kaas.bufferSpec('560', output_size, const=False, ephemeral=True)
    arguments = [(nodes[560], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 561. fused_subtract24
    # kernel 0
    output_size = 1572864
    nodes[561] = kaas.bufferSpec('561', output_size, const=False, ephemeral=True)
    arguments = [(nodes[561], 'o'), (nodes[559], 'i'), (nodes[560], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 562. fused_power_mean24
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a49', output_size, const=False, ephemeral=True))
    arguments = [(nodes[561], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[562] = kaas.bufferSpec('562', output_size, const=False, ephemeral=True)
    arguments = [(nodes[562], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 563. p195
    nodes[563] = addToKV(563, params['p195'])

    # 564. p196
    nodes[564] = addToKV(564, params['p196'])

    # 565. fused_add_sqrt_divide_multiply_add23
    # kernel 0
    output_size = 1572864
    nodes[565] = kaas.bufferSpec('565', output_size, const=False, ephemeral=True)
    arguments = [(nodes[565], 'o'), (nodes[562], 'i'), (nodes[561], 'i'), (nodes[563], 'i'), (nodes[564], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 566. reshape_nop
    nodes[566] = nodes[565]

    # 567. p197
    nodes[567] = addToKV(567, params['p197'])

    # 568. fused_nn_batch_matmul_323
    # kernel 0
    output_size = 1572864
    nodes[568] = kaas.bufferSpec('568', output_size, const=False, ephemeral=True)
    arguments = [(nodes[566], 'i'), (nodes[567], 'i'), (nodes[568], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 569. p198
    nodes[569] = addToKV(569, params['p198'])

    # 570. fused_reshape_add_reshape_transpose_reshape11
    # kernel 0
    output_size = 1572864
    nodes[570] = kaas.bufferSpec('570', output_size, const=False, ephemeral=True)
    arguments = [(nodes[570], 'o'), (nodes[568], 'i'), (nodes[569], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 571. p199
    nodes[571] = addToKV(571, params['p199'])

    # 572. fused_nn_batch_matmul_372
    # kernel 0
    output_size = 1572864
    nodes[572] = kaas.bufferSpec('572', output_size, const=False, ephemeral=True)
    arguments = [(nodes[566], 'i'), (nodes[571], 'i'), (nodes[572], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 573. p200
    nodes[573] = addToKV(573, params['p200'])

    # 574. fused_reshape_add_reshape_transpose_reshape_transpose12
    # kernel 0
    output_size = 1572864
    nodes[574] = kaas.bufferSpec('574', output_size, const=False, ephemeral=True)
    arguments = [(nodes[574], 'o'), (nodes[572], 'i'), (nodes[573], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 575. fused_nn_batch_matmul_511
    # kernel 0
    output_size = 9437184
    nodes[575] = kaas.bufferSpec('575', output_size, const=False, ephemeral=True)
    arguments = [(nodes[570], 'i'), (nodes[574], 'i'), (nodes[575], 'o'), ]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 576. fused_reshape_divide_add11
    # kernel 0
    output_size = 9437184
    nodes[576] = kaas.bufferSpec('576', output_size, const=False, ephemeral=True)
    arguments = [(nodes[576], 'o'), (nodes[575], 'i'), (nodes[23], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 577. fused_max12
    # kernel 0
    output_size = 24576
    nodes[577] = kaas.bufferSpec('577', output_size, const=False, ephemeral=True)
    arguments = [(nodes[576], 'i'), (nodes[577], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 578. fused_subtract_exp11
    # kernel 0
    output_size = 9437184
    nodes[578] = kaas.bufferSpec('578', output_size, const=False, ephemeral=True)
    arguments = [(nodes[578], 'o'), (nodes[576], 'i'), (nodes[577], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 579. fused_sum12
    # kernel 0
    output_size = 24576
    nodes[579] = kaas.bufferSpec('579', output_size, const=False, ephemeral=True)
    arguments = [(nodes[578], 'i'), (nodes[579], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 580. fused_divide_reshape11
    # kernel 0
    output_size = 9437184
    nodes[580] = kaas.bufferSpec('580', output_size, const=False, ephemeral=True)
    arguments = [(nodes[580], 'o'), (nodes[578], 'i'), (nodes[579], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 581. p201
    nodes[581] = addToKV(581, params['p201'])

    # 582. fused_nn_batch_matmul_373
    # kernel 0
    output_size = 1572864
    nodes[582] = kaas.bufferSpec('582', output_size, const=False, ephemeral=True)
    arguments = [(nodes[566], 'i'), (nodes[581], 'i'), (nodes[582], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 583. p202
    nodes[583] = addToKV(583, params['p202'])

    # 584. fused_reshape_add_reshape_transpose_reshape_transpose_112
    # kernel 0
    output_size = 1572864
    nodes[584] = kaas.bufferSpec('584', output_size, const=False, ephemeral=True)
    arguments = [(nodes[584], 'o'), (nodes[582], 'i'), (nodes[583], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 585. fused_nn_batch_matmul_411
    # kernel 0
    output_size = 1572864
    nodes[585] = kaas.bufferSpec('585', output_size, const=False, ephemeral=True)
    arguments = [(nodes[580], 'i'), (nodes[584], 'i'), (nodes[585], 'o'), ]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 586. fused_reshape_transpose_reshape11
    # kernel 0
    output_size = 1572864
    nodes[586] = kaas.bufferSpec('586', output_size, const=False, ephemeral=True)
    arguments = [(nodes[586], 'o'), (nodes[585], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 587. p203
    nodes[587] = addToKV(587, params['p203'])

    # 588. fused_nn_batch_matmul_322
    # kernel 0
    output_size = 1572864
    nodes[588] = kaas.bufferSpec('588', output_size, const=False, ephemeral=True)
    arguments = [(nodes[586], 'i'), (nodes[587], 'i'), (nodes[588], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 589. p204
    nodes[589] = addToKV(589, params['p204'])

    # 590. fused_reshape_add_add23
    # kernel 0
    output_size = 1572864
    nodes[590] = kaas.bufferSpec('590', output_size, const=False, ephemeral=True)
    arguments = [(nodes[590], 'o'), (nodes[588], 'i'), (nodes[589], 'i'), (nodes[565], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 591. fused_mean25
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a50', output_size, const=False, ephemeral=True))
    arguments = [(nodes[590], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[591] = kaas.bufferSpec('591', output_size, const=False, ephemeral=True)
    arguments = [(nodes[591], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 592. fused_subtract23
    # kernel 0
    output_size = 1572864
    nodes[592] = kaas.bufferSpec('592', output_size, const=False, ephemeral=True)
    arguments = [(nodes[592], 'o'), (nodes[590], 'i'), (nodes[591], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 593. fused_power_mean23
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a51', output_size, const=False, ephemeral=True))
    arguments = [(nodes[592], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[593] = kaas.bufferSpec('593', output_size, const=False, ephemeral=True)
    arguments = [(nodes[593], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 594. p205
    nodes[594] = addToKV(594, params['p205'])

    # 595. p206
    nodes[595] = addToKV(595, params['p206'])

    # 596. fused_add_sqrt_divide_multiply_add22
    # kernel 0
    output_size = 1572864
    nodes[596] = kaas.bufferSpec('596', output_size, const=False, ephemeral=True)
    arguments = [(nodes[596], 'o'), (nodes[593], 'i'), (nodes[592], 'i'), (nodes[594], 'i'), (nodes[595], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 597. reshape_nop
    nodes[597] = nodes[596]

    # 598. p207
    nodes[598] = addToKV(598, params['p207'])

    # 599. fused_nn_batch_matmul_211
    # kernel 0
    output_size = 6291456
    nodes[599] = kaas.bufferSpec('599', output_size, const=False, ephemeral=True)
    arguments = [(nodes[597], 'i'), (nodes[598], 'i'), (nodes[599], 'o'), ]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 600. p208
    nodes[600] = addToKV(600, params['p208'])

    # 601. fused_reshape_add_multiply_divide_erf_add_multiply_reshape11
    # kernel 0
    output_size = 6291456
    nodes[601] = kaas.bufferSpec('601', output_size, const=False, ephemeral=True)
    arguments = [(nodes[601], 'o'), (nodes[599], 'i'), (nodes[600], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 602. p209
    nodes[602] = addToKV(602, params['p209'])

    # 603. fused_nn_batch_matmul_111
    # kernel 0
    output_size = 1572864
    nodes[603] = kaas.bufferSpec('603', output_size, const=False, ephemeral=True)
    arguments = [(nodes[601], 'i'), (nodes[602], 'i'), (nodes[603], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 604. p210
    nodes[604] = addToKV(604, params['p210'])

    # 605. fused_reshape_add_add22
    # kernel 0
    output_size = 1572864
    nodes[605] = kaas.bufferSpec('605', output_size, const=False, ephemeral=True)
    arguments = [(nodes[605], 'o'), (nodes[603], 'i'), (nodes[604], 'i'), (nodes[596], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 606. fused_mean26
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a52', output_size, const=False, ephemeral=True))
    arguments = [(nodes[605], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[606] = kaas.bufferSpec('606', output_size, const=False, ephemeral=True)
    arguments = [(nodes[606], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 607. fused_subtract22
    # kernel 0
    output_size = 1572864
    nodes[607] = kaas.bufferSpec('607', output_size, const=False, ephemeral=True)
    arguments = [(nodes[607], 'o'), (nodes[605], 'i'), (nodes[606], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 608. fused_power_mean22
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a53', output_size, const=False, ephemeral=True))
    arguments = [(nodes[607], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[608] = kaas.bufferSpec('608', output_size, const=False, ephemeral=True)
    arguments = [(nodes[608], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 609. p211
    nodes[609] = addToKV(609, params['p211'])

    # 610. p212
    nodes[610] = addToKV(610, params['p212'])

    # 611. fused_add_sqrt_divide_multiply_add21
    # kernel 0
    output_size = 1572864
    nodes[611] = kaas.bufferSpec('611', output_size, const=False, ephemeral=True)
    arguments = [(nodes[611], 'o'), (nodes[608], 'i'), (nodes[607], 'i'), (nodes[609], 'i'), (nodes[610], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 612. reshape_nop
    nodes[612] = nodes[611]

    # 613. p213
    nodes[613] = addToKV(613, params['p213'])

    # 614. fused_nn_batch_matmul_321
    # kernel 0
    output_size = 1572864
    nodes[614] = kaas.bufferSpec('614', output_size, const=False, ephemeral=True)
    arguments = [(nodes[612], 'i'), (nodes[613], 'i'), (nodes[614], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 615. p214
    nodes[615] = addToKV(615, params['p214'])

    # 616. fused_reshape_add_reshape_transpose_reshape10
    # kernel 0
    output_size = 1572864
    nodes[616] = kaas.bufferSpec('616', output_size, const=False, ephemeral=True)
    arguments = [(nodes[616], 'o'), (nodes[614], 'i'), (nodes[615], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 617. p215
    nodes[617] = addToKV(617, params['p215'])

    # 618. fused_nn_batch_matmul_374
    # kernel 0
    output_size = 1572864
    nodes[618] = kaas.bufferSpec('618', output_size, const=False, ephemeral=True)
    arguments = [(nodes[612], 'i'), (nodes[617], 'i'), (nodes[618], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 619. p216
    nodes[619] = addToKV(619, params['p216'])

    # 620. fused_reshape_add_reshape_transpose_reshape_transpose13
    # kernel 0
    output_size = 1572864
    nodes[620] = kaas.bufferSpec('620', output_size, const=False, ephemeral=True)
    arguments = [(nodes[620], 'o'), (nodes[618], 'i'), (nodes[619], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 621. fused_nn_batch_matmul_510
    # kernel 0
    output_size = 9437184
    nodes[621] = kaas.bufferSpec('621', output_size, const=False, ephemeral=True)
    arguments = [(nodes[616], 'i'), (nodes[620], 'i'), (nodes[621], 'o'), ]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 622. fused_reshape_divide_add10
    # kernel 0
    output_size = 9437184
    nodes[622] = kaas.bufferSpec('622', output_size, const=False, ephemeral=True)
    arguments = [(nodes[622], 'o'), (nodes[621], 'i'), (nodes[23], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 623. fused_max13
    # kernel 0
    output_size = 24576
    nodes[623] = kaas.bufferSpec('623', output_size, const=False, ephemeral=True)
    arguments = [(nodes[622], 'i'), (nodes[623], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 624. fused_subtract_exp10
    # kernel 0
    output_size = 9437184
    nodes[624] = kaas.bufferSpec('624', output_size, const=False, ephemeral=True)
    arguments = [(nodes[624], 'o'), (nodes[622], 'i'), (nodes[623], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 625. fused_sum13
    # kernel 0
    output_size = 24576
    nodes[625] = kaas.bufferSpec('625', output_size, const=False, ephemeral=True)
    arguments = [(nodes[624], 'i'), (nodes[625], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 626. fused_divide_reshape10
    # kernel 0
    output_size = 9437184
    nodes[626] = kaas.bufferSpec('626', output_size, const=False, ephemeral=True)
    arguments = [(nodes[626], 'o'), (nodes[624], 'i'), (nodes[625], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 627. p217
    nodes[627] = addToKV(627, params['p217'])

    # 628. fused_nn_batch_matmul_375
    # kernel 0
    output_size = 1572864
    nodes[628] = kaas.bufferSpec('628', output_size, const=False, ephemeral=True)
    arguments = [(nodes[612], 'i'), (nodes[627], 'i'), (nodes[628], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 629. p218
    nodes[629] = addToKV(629, params['p218'])

    # 630. fused_reshape_add_reshape_transpose_reshape_transpose_113
    # kernel 0
    output_size = 1572864
    nodes[630] = kaas.bufferSpec('630', output_size, const=False, ephemeral=True)
    arguments = [(nodes[630], 'o'), (nodes[628], 'i'), (nodes[629], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 631. fused_nn_batch_matmul_410
    # kernel 0
    output_size = 1572864
    nodes[631] = kaas.bufferSpec('631', output_size, const=False, ephemeral=True)
    arguments = [(nodes[626], 'i'), (nodes[630], 'i'), (nodes[631], 'o'), ]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 632. fused_reshape_transpose_reshape10
    # kernel 0
    output_size = 1572864
    nodes[632] = kaas.bufferSpec('632', output_size, const=False, ephemeral=True)
    arguments = [(nodes[632], 'o'), (nodes[631], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 633. p219
    nodes[633] = addToKV(633, params['p219'])

    # 634. fused_nn_batch_matmul_320
    # kernel 0
    output_size = 1572864
    nodes[634] = kaas.bufferSpec('634', output_size, const=False, ephemeral=True)
    arguments = [(nodes[632], 'i'), (nodes[633], 'i'), (nodes[634], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 635. p220
    nodes[635] = addToKV(635, params['p220'])

    # 636. fused_reshape_add_add21
    # kernel 0
    output_size = 1572864
    nodes[636] = kaas.bufferSpec('636', output_size, const=False, ephemeral=True)
    arguments = [(nodes[636], 'o'), (nodes[634], 'i'), (nodes[635], 'i'), (nodes[611], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 637. fused_mean27
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a54', output_size, const=False, ephemeral=True))
    arguments = [(nodes[636], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[637] = kaas.bufferSpec('637', output_size, const=False, ephemeral=True)
    arguments = [(nodes[637], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 638. fused_subtract21
    # kernel 0
    output_size = 1572864
    nodes[638] = kaas.bufferSpec('638', output_size, const=False, ephemeral=True)
    arguments = [(nodes[638], 'o'), (nodes[636], 'i'), (nodes[637], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 639. fused_power_mean21
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a55', output_size, const=False, ephemeral=True))
    arguments = [(nodes[638], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[639] = kaas.bufferSpec('639', output_size, const=False, ephemeral=True)
    arguments = [(nodes[639], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 640. p221
    nodes[640] = addToKV(640, params['p221'])

    # 641. p222
    nodes[641] = addToKV(641, params['p222'])

    # 642. fused_add_sqrt_divide_multiply_add20
    # kernel 0
    output_size = 1572864
    nodes[642] = kaas.bufferSpec('642', output_size, const=False, ephemeral=True)
    arguments = [(nodes[642], 'o'), (nodes[639], 'i'), (nodes[638], 'i'), (nodes[640], 'i'), (nodes[641], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 643. reshape_nop
    nodes[643] = nodes[642]

    # 644. p223
    nodes[644] = addToKV(644, params['p223'])

    # 645. fused_nn_batch_matmul_210
    # kernel 0
    output_size = 6291456
    nodes[645] = kaas.bufferSpec('645', output_size, const=False, ephemeral=True)
    arguments = [(nodes[643], 'i'), (nodes[644], 'i'), (nodes[645], 'o'), ]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 646. p224
    nodes[646] = addToKV(646, params['p224'])

    # 647. fused_reshape_add_multiply_divide_erf_add_multiply_reshape10
    # kernel 0
    output_size = 6291456
    nodes[647] = kaas.bufferSpec('647', output_size, const=False, ephemeral=True)
    arguments = [(nodes[647], 'o'), (nodes[645], 'i'), (nodes[646], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 648. p225
    nodes[648] = addToKV(648, params['p225'])

    # 649. fused_nn_batch_matmul_110
    # kernel 0
    output_size = 1572864
    nodes[649] = kaas.bufferSpec('649', output_size, const=False, ephemeral=True)
    arguments = [(nodes[647], 'i'), (nodes[648], 'i'), (nodes[649], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 650. p226
    nodes[650] = addToKV(650, params['p226'])

    # 651. fused_reshape_add_add20
    # kernel 0
    output_size = 1572864
    nodes[651] = kaas.bufferSpec('651', output_size, const=False, ephemeral=True)
    arguments = [(nodes[651], 'o'), (nodes[649], 'i'), (nodes[650], 'i'), (nodes[642], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 652. fused_mean28
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a56', output_size, const=False, ephemeral=True))
    arguments = [(nodes[651], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[652] = kaas.bufferSpec('652', output_size, const=False, ephemeral=True)
    arguments = [(nodes[652], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 653. fused_subtract20
    # kernel 0
    output_size = 1572864
    nodes[653] = kaas.bufferSpec('653', output_size, const=False, ephemeral=True)
    arguments = [(nodes[653], 'o'), (nodes[651], 'i'), (nodes[652], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 654. fused_power_mean20
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a57', output_size, const=False, ephemeral=True))
    arguments = [(nodes[653], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[654] = kaas.bufferSpec('654', output_size, const=False, ephemeral=True)
    arguments = [(nodes[654], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 655. p227
    nodes[655] = addToKV(655, params['p227'])

    # 656. p228
    nodes[656] = addToKV(656, params['p228'])

    # 657. fused_add_sqrt_divide_multiply_add19
    # kernel 0
    output_size = 1572864
    nodes[657] = kaas.bufferSpec('657', output_size, const=False, ephemeral=True)
    arguments = [(nodes[657], 'o'), (nodes[654], 'i'), (nodes[653], 'i'), (nodes[655], 'i'), (nodes[656], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 658. reshape_nop
    nodes[658] = nodes[657]

    # 659. p229
    nodes[659] = addToKV(659, params['p229'])

    # 660. fused_nn_batch_matmul_319
    # kernel 0
    output_size = 1572864
    nodes[660] = kaas.bufferSpec('660', output_size, const=False, ephemeral=True)
    arguments = [(nodes[658], 'i'), (nodes[659], 'i'), (nodes[660], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 661. p230
    nodes[661] = addToKV(661, params['p230'])

    # 662. fused_reshape_add_reshape_transpose_reshape9
    # kernel 0
    output_size = 1572864
    nodes[662] = kaas.bufferSpec('662', output_size, const=False, ephemeral=True)
    arguments = [(nodes[662], 'o'), (nodes[660], 'i'), (nodes[661], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 663. p231
    nodes[663] = addToKV(663, params['p231'])

    # 664. fused_nn_batch_matmul_376
    # kernel 0
    output_size = 1572864
    nodes[664] = kaas.bufferSpec('664', output_size, const=False, ephemeral=True)
    arguments = [(nodes[658], 'i'), (nodes[663], 'i'), (nodes[664], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 665. p232
    nodes[665] = addToKV(665, params['p232'])

    # 666. fused_reshape_add_reshape_transpose_reshape_transpose14
    # kernel 0
    output_size = 1572864
    nodes[666] = kaas.bufferSpec('666', output_size, const=False, ephemeral=True)
    arguments = [(nodes[666], 'o'), (nodes[664], 'i'), (nodes[665], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 667. fused_nn_batch_matmul_59
    # kernel 0
    output_size = 9437184
    nodes[667] = kaas.bufferSpec('667', output_size, const=False, ephemeral=True)
    arguments = [(nodes[662], 'i'), (nodes[666], 'i'), (nodes[667], 'o'), ]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 668. fused_reshape_divide_add9
    # kernel 0
    output_size = 9437184
    nodes[668] = kaas.bufferSpec('668', output_size, const=False, ephemeral=True)
    arguments = [(nodes[668], 'o'), (nodes[667], 'i'), (nodes[23], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 669. fused_max14
    # kernel 0
    output_size = 24576
    nodes[669] = kaas.bufferSpec('669', output_size, const=False, ephemeral=True)
    arguments = [(nodes[668], 'i'), (nodes[669], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 670. fused_subtract_exp9
    # kernel 0
    output_size = 9437184
    nodes[670] = kaas.bufferSpec('670', output_size, const=False, ephemeral=True)
    arguments = [(nodes[670], 'o'), (nodes[668], 'i'), (nodes[669], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 671. fused_sum14
    # kernel 0
    output_size = 24576
    nodes[671] = kaas.bufferSpec('671', output_size, const=False, ephemeral=True)
    arguments = [(nodes[670], 'i'), (nodes[671], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 672. fused_divide_reshape9
    # kernel 0
    output_size = 9437184
    nodes[672] = kaas.bufferSpec('672', output_size, const=False, ephemeral=True)
    arguments = [(nodes[672], 'o'), (nodes[670], 'i'), (nodes[671], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 673. p233
    nodes[673] = addToKV(673, params['p233'])

    # 674. fused_nn_batch_matmul_377
    # kernel 0
    output_size = 1572864
    nodes[674] = kaas.bufferSpec('674', output_size, const=False, ephemeral=True)
    arguments = [(nodes[658], 'i'), (nodes[673], 'i'), (nodes[674], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 675. p234
    nodes[675] = addToKV(675, params['p234'])

    # 676. fused_reshape_add_reshape_transpose_reshape_transpose_114
    # kernel 0
    output_size = 1572864
    nodes[676] = kaas.bufferSpec('676', output_size, const=False, ephemeral=True)
    arguments = [(nodes[676], 'o'), (nodes[674], 'i'), (nodes[675], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 677. fused_nn_batch_matmul_49
    # kernel 0
    output_size = 1572864
    nodes[677] = kaas.bufferSpec('677', output_size, const=False, ephemeral=True)
    arguments = [(nodes[672], 'i'), (nodes[676], 'i'), (nodes[677], 'o'), ]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 678. fused_reshape_transpose_reshape9
    # kernel 0
    output_size = 1572864
    nodes[678] = kaas.bufferSpec('678', output_size, const=False, ephemeral=True)
    arguments = [(nodes[678], 'o'), (nodes[677], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 679. p235
    nodes[679] = addToKV(679, params['p235'])

    # 680. fused_nn_batch_matmul_318
    # kernel 0
    output_size = 1572864
    nodes[680] = kaas.bufferSpec('680', output_size, const=False, ephemeral=True)
    arguments = [(nodes[678], 'i'), (nodes[679], 'i'), (nodes[680], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 681. p236
    nodes[681] = addToKV(681, params['p236'])

    # 682. fused_reshape_add_add19
    # kernel 0
    output_size = 1572864
    nodes[682] = kaas.bufferSpec('682', output_size, const=False, ephemeral=True)
    arguments = [(nodes[682], 'o'), (nodes[680], 'i'), (nodes[681], 'i'), (nodes[657], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 683. fused_mean29
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a58', output_size, const=False, ephemeral=True))
    arguments = [(nodes[682], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[683] = kaas.bufferSpec('683', output_size, const=False, ephemeral=True)
    arguments = [(nodes[683], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 684. fused_subtract19
    # kernel 0
    output_size = 1572864
    nodes[684] = kaas.bufferSpec('684', output_size, const=False, ephemeral=True)
    arguments = [(nodes[684], 'o'), (nodes[682], 'i'), (nodes[683], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 685. fused_power_mean19
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a59', output_size, const=False, ephemeral=True))
    arguments = [(nodes[684], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[685] = kaas.bufferSpec('685', output_size, const=False, ephemeral=True)
    arguments = [(nodes[685], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 686. p237
    nodes[686] = addToKV(686, params['p237'])

    # 687. p238
    nodes[687] = addToKV(687, params['p238'])

    # 688. fused_add_sqrt_divide_multiply_add18
    # kernel 0
    output_size = 1572864
    nodes[688] = kaas.bufferSpec('688', output_size, const=False, ephemeral=True)
    arguments = [(nodes[688], 'o'), (nodes[685], 'i'), (nodes[684], 'i'), (nodes[686], 'i'), (nodes[687], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 689. reshape_nop
    nodes[689] = nodes[688]

    # 690. p239
    nodes[690] = addToKV(690, params['p239'])

    # 691. fused_nn_batch_matmul_29
    # kernel 0
    output_size = 6291456
    nodes[691] = kaas.bufferSpec('691', output_size, const=False, ephemeral=True)
    arguments = [(nodes[689], 'i'), (nodes[690], 'i'), (nodes[691], 'o'), ]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 692. p240
    nodes[692] = addToKV(692, params['p240'])

    # 693. fused_reshape_add_multiply_divide_erf_add_multiply_reshape9
    # kernel 0
    output_size = 6291456
    nodes[693] = kaas.bufferSpec('693', output_size, const=False, ephemeral=True)
    arguments = [(nodes[693], 'o'), (nodes[691], 'i'), (nodes[692], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 694. p241
    nodes[694] = addToKV(694, params['p241'])

    # 695. fused_nn_batch_matmul_19
    # kernel 0
    output_size = 1572864
    nodes[695] = kaas.bufferSpec('695', output_size, const=False, ephemeral=True)
    arguments = [(nodes[693], 'i'), (nodes[694], 'i'), (nodes[695], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 696. p242
    nodes[696] = addToKV(696, params['p242'])

    # 697. fused_reshape_add_add18
    # kernel 0
    output_size = 1572864
    nodes[697] = kaas.bufferSpec('697', output_size, const=False, ephemeral=True)
    arguments = [(nodes[697], 'o'), (nodes[695], 'i'), (nodes[696], 'i'), (nodes[688], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 698. fused_mean30
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a60', output_size, const=False, ephemeral=True))
    arguments = [(nodes[697], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[698] = kaas.bufferSpec('698', output_size, const=False, ephemeral=True)
    arguments = [(nodes[698], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 699. fused_subtract18
    # kernel 0
    output_size = 1572864
    nodes[699] = kaas.bufferSpec('699', output_size, const=False, ephemeral=True)
    arguments = [(nodes[699], 'o'), (nodes[697], 'i'), (nodes[698], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 700. fused_power_mean18
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a61', output_size, const=False, ephemeral=True))
    arguments = [(nodes[699], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[700] = kaas.bufferSpec('700', output_size, const=False, ephemeral=True)
    arguments = [(nodes[700], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 701. p243
    nodes[701] = addToKV(701, params['p243'])

    # 702. p244
    nodes[702] = addToKV(702, params['p244'])

    # 703. fused_add_sqrt_divide_multiply_add17
    # kernel 0
    output_size = 1572864
    nodes[703] = kaas.bufferSpec('703', output_size, const=False, ephemeral=True)
    arguments = [(nodes[703], 'o'), (nodes[700], 'i'), (nodes[699], 'i'), (nodes[701], 'i'), (nodes[702], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 704. reshape_nop
    nodes[704] = nodes[703]

    # 705. p245
    nodes[705] = addToKV(705, params['p245'])

    # 706. fused_nn_batch_matmul_317
    # kernel 0
    output_size = 1572864
    nodes[706] = kaas.bufferSpec('706', output_size, const=False, ephemeral=True)
    arguments = [(nodes[704], 'i'), (nodes[705], 'i'), (nodes[706], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 707. p246
    nodes[707] = addToKV(707, params['p246'])

    # 708. fused_reshape_add_reshape_transpose_reshape8
    # kernel 0
    output_size = 1572864
    nodes[708] = kaas.bufferSpec('708', output_size, const=False, ephemeral=True)
    arguments = [(nodes[708], 'o'), (nodes[706], 'i'), (nodes[707], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 709. p247
    nodes[709] = addToKV(709, params['p247'])

    # 710. fused_nn_batch_matmul_378
    # kernel 0
    output_size = 1572864
    nodes[710] = kaas.bufferSpec('710', output_size, const=False, ephemeral=True)
    arguments = [(nodes[704], 'i'), (nodes[709], 'i'), (nodes[710], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 711. p248
    nodes[711] = addToKV(711, params['p248'])

    # 712. fused_reshape_add_reshape_transpose_reshape_transpose15
    # kernel 0
    output_size = 1572864
    nodes[712] = kaas.bufferSpec('712', output_size, const=False, ephemeral=True)
    arguments = [(nodes[712], 'o'), (nodes[710], 'i'), (nodes[711], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 713. fused_nn_batch_matmul_58
    # kernel 0
    output_size = 9437184
    nodes[713] = kaas.bufferSpec('713', output_size, const=False, ephemeral=True)
    arguments = [(nodes[708], 'i'), (nodes[712], 'i'), (nodes[713], 'o'), ]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 714. fused_reshape_divide_add8
    # kernel 0
    output_size = 9437184
    nodes[714] = kaas.bufferSpec('714', output_size, const=False, ephemeral=True)
    arguments = [(nodes[714], 'o'), (nodes[713], 'i'), (nodes[23], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 715. fused_max15
    # kernel 0
    output_size = 24576
    nodes[715] = kaas.bufferSpec('715', output_size, const=False, ephemeral=True)
    arguments = [(nodes[714], 'i'), (nodes[715], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 716. fused_subtract_exp8
    # kernel 0
    output_size = 9437184
    nodes[716] = kaas.bufferSpec('716', output_size, const=False, ephemeral=True)
    arguments = [(nodes[716], 'o'), (nodes[714], 'i'), (nodes[715], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 717. fused_sum15
    # kernel 0
    output_size = 24576
    nodes[717] = kaas.bufferSpec('717', output_size, const=False, ephemeral=True)
    arguments = [(nodes[716], 'i'), (nodes[717], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 718. fused_divide_reshape8
    # kernel 0
    output_size = 9437184
    nodes[718] = kaas.bufferSpec('718', output_size, const=False, ephemeral=True)
    arguments = [(nodes[718], 'o'), (nodes[716], 'i'), (nodes[717], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 719. p249
    nodes[719] = addToKV(719, params['p249'])

    # 720. fused_nn_batch_matmul_379
    # kernel 0
    output_size = 1572864
    nodes[720] = kaas.bufferSpec('720', output_size, const=False, ephemeral=True)
    arguments = [(nodes[704], 'i'), (nodes[719], 'i'), (nodes[720], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 721. p250
    nodes[721] = addToKV(721, params['p250'])

    # 722. fused_reshape_add_reshape_transpose_reshape_transpose_115
    # kernel 0
    output_size = 1572864
    nodes[722] = kaas.bufferSpec('722', output_size, const=False, ephemeral=True)
    arguments = [(nodes[722], 'o'), (nodes[720], 'i'), (nodes[721], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 723. fused_nn_batch_matmul_48
    # kernel 0
    output_size = 1572864
    nodes[723] = kaas.bufferSpec('723', output_size, const=False, ephemeral=True)
    arguments = [(nodes[718], 'i'), (nodes[722], 'i'), (nodes[723], 'o'), ]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 724. fused_reshape_transpose_reshape8
    # kernel 0
    output_size = 1572864
    nodes[724] = kaas.bufferSpec('724', output_size, const=False, ephemeral=True)
    arguments = [(nodes[724], 'o'), (nodes[723], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 725. p251
    nodes[725] = addToKV(725, params['p251'])

    # 726. fused_nn_batch_matmul_316
    # kernel 0
    output_size = 1572864
    nodes[726] = kaas.bufferSpec('726', output_size, const=False, ephemeral=True)
    arguments = [(nodes[724], 'i'), (nodes[725], 'i'), (nodes[726], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 727. p252
    nodes[727] = addToKV(727, params['p252'])

    # 728. fused_reshape_add_add17
    # kernel 0
    output_size = 1572864
    nodes[728] = kaas.bufferSpec('728', output_size, const=False, ephemeral=True)
    arguments = [(nodes[728], 'o'), (nodes[726], 'i'), (nodes[727], 'i'), (nodes[703], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 729. fused_mean31
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a62', output_size, const=False, ephemeral=True))
    arguments = [(nodes[728], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[729] = kaas.bufferSpec('729', output_size, const=False, ephemeral=True)
    arguments = [(nodes[729], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 730. fused_subtract17
    # kernel 0
    output_size = 1572864
    nodes[730] = kaas.bufferSpec('730', output_size, const=False, ephemeral=True)
    arguments = [(nodes[730], 'o'), (nodes[728], 'i'), (nodes[729], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 731. fused_power_mean17
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a63', output_size, const=False, ephemeral=True))
    arguments = [(nodes[730], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[731] = kaas.bufferSpec('731', output_size, const=False, ephemeral=True)
    arguments = [(nodes[731], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 732. p253
    nodes[732] = addToKV(732, params['p253'])

    # 733. p254
    nodes[733] = addToKV(733, params['p254'])

    # 734. fused_add_sqrt_divide_multiply_add16
    # kernel 0
    output_size = 1572864
    nodes[734] = kaas.bufferSpec('734', output_size, const=False, ephemeral=True)
    arguments = [(nodes[734], 'o'), (nodes[731], 'i'), (nodes[730], 'i'), (nodes[732], 'i'), (nodes[733], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 735. reshape_nop
    nodes[735] = nodes[734]

    # 736. p255
    nodes[736] = addToKV(736, params['p255'])

    # 737. fused_nn_batch_matmul_28
    # kernel 0
    output_size = 6291456
    nodes[737] = kaas.bufferSpec('737', output_size, const=False, ephemeral=True)
    arguments = [(nodes[735], 'i'), (nodes[736], 'i'), (nodes[737], 'o'), ]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 738. p256
    nodes[738] = addToKV(738, params['p256'])

    # 739. fused_reshape_add_multiply_divide_erf_add_multiply_reshape8
    # kernel 0
    output_size = 6291456
    nodes[739] = kaas.bufferSpec('739', output_size, const=False, ephemeral=True)
    arguments = [(nodes[739], 'o'), (nodes[737], 'i'), (nodes[738], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 740. p257
    nodes[740] = addToKV(740, params['p257'])

    # 741. fused_nn_batch_matmul_18
    # kernel 0
    output_size = 1572864
    nodes[741] = kaas.bufferSpec('741', output_size, const=False, ephemeral=True)
    arguments = [(nodes[739], 'i'), (nodes[740], 'i'), (nodes[741], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 742. p258
    nodes[742] = addToKV(742, params['p258'])

    # 743. fused_reshape_add_add16
    # kernel 0
    output_size = 1572864
    nodes[743] = kaas.bufferSpec('743', output_size, const=False, ephemeral=True)
    arguments = [(nodes[743], 'o'), (nodes[741], 'i'), (nodes[742], 'i'), (nodes[734], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 744. fused_mean32
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a64', output_size, const=False, ephemeral=True))
    arguments = [(nodes[743], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[744] = kaas.bufferSpec('744', output_size, const=False, ephemeral=True)
    arguments = [(nodes[744], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 745. fused_subtract16
    # kernel 0
    output_size = 1572864
    nodes[745] = kaas.bufferSpec('745', output_size, const=False, ephemeral=True)
    arguments = [(nodes[745], 'o'), (nodes[743], 'i'), (nodes[744], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 746. fused_power_mean16
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a65', output_size, const=False, ephemeral=True))
    arguments = [(nodes[745], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[746] = kaas.bufferSpec('746', output_size, const=False, ephemeral=True)
    arguments = [(nodes[746], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 747. p259
    nodes[747] = addToKV(747, params['p259'])

    # 748. p260
    nodes[748] = addToKV(748, params['p260'])

    # 749. fused_add_sqrt_divide_multiply_add15
    # kernel 0
    output_size = 1572864
    nodes[749] = kaas.bufferSpec('749', output_size, const=False, ephemeral=True)
    arguments = [(nodes[749], 'o'), (nodes[746], 'i'), (nodes[745], 'i'), (nodes[747], 'i'), (nodes[748], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 750. reshape_nop
    nodes[750] = nodes[749]

    # 751. p261
    nodes[751] = addToKV(751, params['p261'])

    # 752. fused_nn_batch_matmul_315
    # kernel 0
    output_size = 1572864
    nodes[752] = kaas.bufferSpec('752', output_size, const=False, ephemeral=True)
    arguments = [(nodes[750], 'i'), (nodes[751], 'i'), (nodes[752], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 753. p262
    nodes[753] = addToKV(753, params['p262'])

    # 754. fused_reshape_add_reshape_transpose_reshape7
    # kernel 0
    output_size = 1572864
    nodes[754] = kaas.bufferSpec('754', output_size, const=False, ephemeral=True)
    arguments = [(nodes[754], 'o'), (nodes[752], 'i'), (nodes[753], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 755. p263
    nodes[755] = addToKV(755, params['p263'])

    # 756. fused_nn_batch_matmul_380
    # kernel 0
    output_size = 1572864
    nodes[756] = kaas.bufferSpec('756', output_size, const=False, ephemeral=True)
    arguments = [(nodes[750], 'i'), (nodes[755], 'i'), (nodes[756], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 757. p264
    nodes[757] = addToKV(757, params['p264'])

    # 758. fused_reshape_add_reshape_transpose_reshape_transpose16
    # kernel 0
    output_size = 1572864
    nodes[758] = kaas.bufferSpec('758', output_size, const=False, ephemeral=True)
    arguments = [(nodes[758], 'o'), (nodes[756], 'i'), (nodes[757], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 759. fused_nn_batch_matmul_57
    # kernel 0
    output_size = 9437184
    nodes[759] = kaas.bufferSpec('759', output_size, const=False, ephemeral=True)
    arguments = [(nodes[754], 'i'), (nodes[758], 'i'), (nodes[759], 'o'), ]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 760. fused_reshape_divide_add7
    # kernel 0
    output_size = 9437184
    nodes[760] = kaas.bufferSpec('760', output_size, const=False, ephemeral=True)
    arguments = [(nodes[760], 'o'), (nodes[759], 'i'), (nodes[23], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 761. fused_max16
    # kernel 0
    output_size = 24576
    nodes[761] = kaas.bufferSpec('761', output_size, const=False, ephemeral=True)
    arguments = [(nodes[760], 'i'), (nodes[761], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 762. fused_subtract_exp7
    # kernel 0
    output_size = 9437184
    nodes[762] = kaas.bufferSpec('762', output_size, const=False, ephemeral=True)
    arguments = [(nodes[762], 'o'), (nodes[760], 'i'), (nodes[761], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 763. fused_sum16
    # kernel 0
    output_size = 24576
    nodes[763] = kaas.bufferSpec('763', output_size, const=False, ephemeral=True)
    arguments = [(nodes[762], 'i'), (nodes[763], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 764. fused_divide_reshape7
    # kernel 0
    output_size = 9437184
    nodes[764] = kaas.bufferSpec('764', output_size, const=False, ephemeral=True)
    arguments = [(nodes[764], 'o'), (nodes[762], 'i'), (nodes[763], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 765. p265
    nodes[765] = addToKV(765, params['p265'])

    # 766. fused_nn_batch_matmul_381
    # kernel 0
    output_size = 1572864
    nodes[766] = kaas.bufferSpec('766', output_size, const=False, ephemeral=True)
    arguments = [(nodes[750], 'i'), (nodes[765], 'i'), (nodes[766], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 767. p266
    nodes[767] = addToKV(767, params['p266'])

    # 768. fused_reshape_add_reshape_transpose_reshape_transpose_116
    # kernel 0
    output_size = 1572864
    nodes[768] = kaas.bufferSpec('768', output_size, const=False, ephemeral=True)
    arguments = [(nodes[768], 'o'), (nodes[766], 'i'), (nodes[767], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 769. fused_nn_batch_matmul_47
    # kernel 0
    output_size = 1572864
    nodes[769] = kaas.bufferSpec('769', output_size, const=False, ephemeral=True)
    arguments = [(nodes[764], 'i'), (nodes[768], 'i'), (nodes[769], 'o'), ]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 770. fused_reshape_transpose_reshape7
    # kernel 0
    output_size = 1572864
    nodes[770] = kaas.bufferSpec('770', output_size, const=False, ephemeral=True)
    arguments = [(nodes[770], 'o'), (nodes[769], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 771. p267
    nodes[771] = addToKV(771, params['p267'])

    # 772. fused_nn_batch_matmul_314
    # kernel 0
    output_size = 1572864
    nodes[772] = kaas.bufferSpec('772', output_size, const=False, ephemeral=True)
    arguments = [(nodes[770], 'i'), (nodes[771], 'i'), (nodes[772], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 773. p268
    nodes[773] = addToKV(773, params['p268'])

    # 774. fused_reshape_add_add15
    # kernel 0
    output_size = 1572864
    nodes[774] = kaas.bufferSpec('774', output_size, const=False, ephemeral=True)
    arguments = [(nodes[774], 'o'), (nodes[772], 'i'), (nodes[773], 'i'), (nodes[749], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 775. fused_mean33
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a66', output_size, const=False, ephemeral=True))
    arguments = [(nodes[774], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[775] = kaas.bufferSpec('775', output_size, const=False, ephemeral=True)
    arguments = [(nodes[775], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 776. fused_subtract15
    # kernel 0
    output_size = 1572864
    nodes[776] = kaas.bufferSpec('776', output_size, const=False, ephemeral=True)
    arguments = [(nodes[776], 'o'), (nodes[774], 'i'), (nodes[775], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 777. fused_power_mean15
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a67', output_size, const=False, ephemeral=True))
    arguments = [(nodes[776], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[777] = kaas.bufferSpec('777', output_size, const=False, ephemeral=True)
    arguments = [(nodes[777], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 778. p269
    nodes[778] = addToKV(778, params['p269'])

    # 779. p270
    nodes[779] = addToKV(779, params['p270'])

    # 780. fused_add_sqrt_divide_multiply_add14
    # kernel 0
    output_size = 1572864
    nodes[780] = kaas.bufferSpec('780', output_size, const=False, ephemeral=True)
    arguments = [(nodes[780], 'o'), (nodes[777], 'i'), (nodes[776], 'i'), (nodes[778], 'i'), (nodes[779], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 781. reshape_nop
    nodes[781] = nodes[780]

    # 782. p271
    nodes[782] = addToKV(782, params['p271'])

    # 783. fused_nn_batch_matmul_27
    # kernel 0
    output_size = 6291456
    nodes[783] = kaas.bufferSpec('783', output_size, const=False, ephemeral=True)
    arguments = [(nodes[781], 'i'), (nodes[782], 'i'), (nodes[783], 'o'), ]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 784. p272
    nodes[784] = addToKV(784, params['p272'])

    # 785. fused_reshape_add_multiply_divide_erf_add_multiply_reshape7
    # kernel 0
    output_size = 6291456
    nodes[785] = kaas.bufferSpec('785', output_size, const=False, ephemeral=True)
    arguments = [(nodes[785], 'o'), (nodes[783], 'i'), (nodes[784], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 786. p273
    nodes[786] = addToKV(786, params['p273'])

    # 787. fused_nn_batch_matmul_17
    # kernel 0
    output_size = 1572864
    nodes[787] = kaas.bufferSpec('787', output_size, const=False, ephemeral=True)
    arguments = [(nodes[785], 'i'), (nodes[786], 'i'), (nodes[787], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 788. p274
    nodes[788] = addToKV(788, params['p274'])

    # 789. fused_reshape_add_add14
    # kernel 0
    output_size = 1572864
    nodes[789] = kaas.bufferSpec('789', output_size, const=False, ephemeral=True)
    arguments = [(nodes[789], 'o'), (nodes[787], 'i'), (nodes[788], 'i'), (nodes[780], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 790. fused_mean34
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a68', output_size, const=False, ephemeral=True))
    arguments = [(nodes[789], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[790] = kaas.bufferSpec('790', output_size, const=False, ephemeral=True)
    arguments = [(nodes[790], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 791. fused_subtract14
    # kernel 0
    output_size = 1572864
    nodes[791] = kaas.bufferSpec('791', output_size, const=False, ephemeral=True)
    arguments = [(nodes[791], 'o'), (nodes[789], 'i'), (nodes[790], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 792. fused_power_mean14
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a69', output_size, const=False, ephemeral=True))
    arguments = [(nodes[791], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[792] = kaas.bufferSpec('792', output_size, const=False, ephemeral=True)
    arguments = [(nodes[792], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 793. p275
    nodes[793] = addToKV(793, params['p275'])

    # 794. p276
    nodes[794] = addToKV(794, params['p276'])

    # 795. fused_add_sqrt_divide_multiply_add13
    # kernel 0
    output_size = 1572864
    nodes[795] = kaas.bufferSpec('795', output_size, const=False, ephemeral=True)
    arguments = [(nodes[795], 'o'), (nodes[792], 'i'), (nodes[791], 'i'), (nodes[793], 'i'), (nodes[794], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 796. reshape_nop
    nodes[796] = nodes[795]

    # 797. p277
    nodes[797] = addToKV(797, params['p277'])

    # 798. fused_nn_batch_matmul_313
    # kernel 0
    output_size = 1572864
    nodes[798] = kaas.bufferSpec('798', output_size, const=False, ephemeral=True)
    arguments = [(nodes[796], 'i'), (nodes[797], 'i'), (nodes[798], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 799. p278
    nodes[799] = addToKV(799, params['p278'])

    # 800. fused_reshape_add_reshape_transpose_reshape6
    # kernel 0
    output_size = 1572864
    nodes[800] = kaas.bufferSpec('800', output_size, const=False, ephemeral=True)
    arguments = [(nodes[800], 'o'), (nodes[798], 'i'), (nodes[799], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 801. p279
    nodes[801] = addToKV(801, params['p279'])

    # 802. fused_nn_batch_matmul_382
    # kernel 0
    output_size = 1572864
    nodes[802] = kaas.bufferSpec('802', output_size, const=False, ephemeral=True)
    arguments = [(nodes[796], 'i'), (nodes[801], 'i'), (nodes[802], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 803. p280
    nodes[803] = addToKV(803, params['p280'])

    # 804. fused_reshape_add_reshape_transpose_reshape_transpose17
    # kernel 0
    output_size = 1572864
    nodes[804] = kaas.bufferSpec('804', output_size, const=False, ephemeral=True)
    arguments = [(nodes[804], 'o'), (nodes[802], 'i'), (nodes[803], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 805. fused_nn_batch_matmul_56
    # kernel 0
    output_size = 9437184
    nodes[805] = kaas.bufferSpec('805', output_size, const=False, ephemeral=True)
    arguments = [(nodes[800], 'i'), (nodes[804], 'i'), (nodes[805], 'o'), ]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 806. fused_reshape_divide_add6
    # kernel 0
    output_size = 9437184
    nodes[806] = kaas.bufferSpec('806', output_size, const=False, ephemeral=True)
    arguments = [(nodes[806], 'o'), (nodes[805], 'i'), (nodes[23], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 807. fused_max17
    # kernel 0
    output_size = 24576
    nodes[807] = kaas.bufferSpec('807', output_size, const=False, ephemeral=True)
    arguments = [(nodes[806], 'i'), (nodes[807], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 808. fused_subtract_exp6
    # kernel 0
    output_size = 9437184
    nodes[808] = kaas.bufferSpec('808', output_size, const=False, ephemeral=True)
    arguments = [(nodes[808], 'o'), (nodes[806], 'i'), (nodes[807], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 809. fused_sum17
    # kernel 0
    output_size = 24576
    nodes[809] = kaas.bufferSpec('809', output_size, const=False, ephemeral=True)
    arguments = [(nodes[808], 'i'), (nodes[809], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 810. fused_divide_reshape6
    # kernel 0
    output_size = 9437184
    nodes[810] = kaas.bufferSpec('810', output_size, const=False, ephemeral=True)
    arguments = [(nodes[810], 'o'), (nodes[808], 'i'), (nodes[809], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 811. p281
    nodes[811] = addToKV(811, params['p281'])

    # 812. fused_nn_batch_matmul_383
    # kernel 0
    output_size = 1572864
    nodes[812] = kaas.bufferSpec('812', output_size, const=False, ephemeral=True)
    arguments = [(nodes[796], 'i'), (nodes[811], 'i'), (nodes[812], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 813. p282
    nodes[813] = addToKV(813, params['p282'])

    # 814. fused_reshape_add_reshape_transpose_reshape_transpose_117
    # kernel 0
    output_size = 1572864
    nodes[814] = kaas.bufferSpec('814', output_size, const=False, ephemeral=True)
    arguments = [(nodes[814], 'o'), (nodes[812], 'i'), (nodes[813], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 815. fused_nn_batch_matmul_46
    # kernel 0
    output_size = 1572864
    nodes[815] = kaas.bufferSpec('815', output_size, const=False, ephemeral=True)
    arguments = [(nodes[810], 'i'), (nodes[814], 'i'), (nodes[815], 'o'), ]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 816. fused_reshape_transpose_reshape6
    # kernel 0
    output_size = 1572864
    nodes[816] = kaas.bufferSpec('816', output_size, const=False, ephemeral=True)
    arguments = [(nodes[816], 'o'), (nodes[815], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 817. p283
    nodes[817] = addToKV(817, params['p283'])

    # 818. fused_nn_batch_matmul_312
    # kernel 0
    output_size = 1572864
    nodes[818] = kaas.bufferSpec('818', output_size, const=False, ephemeral=True)
    arguments = [(nodes[816], 'i'), (nodes[817], 'i'), (nodes[818], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 819. p284
    nodes[819] = addToKV(819, params['p284'])

    # 820. fused_reshape_add_add13
    # kernel 0
    output_size = 1572864
    nodes[820] = kaas.bufferSpec('820', output_size, const=False, ephemeral=True)
    arguments = [(nodes[820], 'o'), (nodes[818], 'i'), (nodes[819], 'i'), (nodes[795], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 821. fused_mean35
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a70', output_size, const=False, ephemeral=True))
    arguments = [(nodes[820], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[821] = kaas.bufferSpec('821', output_size, const=False, ephemeral=True)
    arguments = [(nodes[821], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 822. fused_subtract13
    # kernel 0
    output_size = 1572864
    nodes[822] = kaas.bufferSpec('822', output_size, const=False, ephemeral=True)
    arguments = [(nodes[822], 'o'), (nodes[820], 'i'), (nodes[821], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 823. fused_power_mean13
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a71', output_size, const=False, ephemeral=True))
    arguments = [(nodes[822], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[823] = kaas.bufferSpec('823', output_size, const=False, ephemeral=True)
    arguments = [(nodes[823], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 824. p285
    nodes[824] = addToKV(824, params['p285'])

    # 825. p286
    nodes[825] = addToKV(825, params['p286'])

    # 826. fused_add_sqrt_divide_multiply_add12
    # kernel 0
    output_size = 1572864
    nodes[826] = kaas.bufferSpec('826', output_size, const=False, ephemeral=True)
    arguments = [(nodes[826], 'o'), (nodes[823], 'i'), (nodes[822], 'i'), (nodes[824], 'i'), (nodes[825], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 827. reshape_nop
    nodes[827] = nodes[826]

    # 828. p287
    nodes[828] = addToKV(828, params['p287'])

    # 829. fused_nn_batch_matmul_26
    # kernel 0
    output_size = 6291456
    nodes[829] = kaas.bufferSpec('829', output_size, const=False, ephemeral=True)
    arguments = [(nodes[827], 'i'), (nodes[828], 'i'), (nodes[829], 'o'), ]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 830. p288
    nodes[830] = addToKV(830, params['p288'])

    # 831. fused_reshape_add_multiply_divide_erf_add_multiply_reshape6
    # kernel 0
    output_size = 6291456
    nodes[831] = kaas.bufferSpec('831', output_size, const=False, ephemeral=True)
    arguments = [(nodes[831], 'o'), (nodes[829], 'i'), (nodes[830], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 832. p289
    nodes[832] = addToKV(832, params['p289'])

    # 833. fused_nn_batch_matmul_16
    # kernel 0
    output_size = 1572864
    nodes[833] = kaas.bufferSpec('833', output_size, const=False, ephemeral=True)
    arguments = [(nodes[831], 'i'), (nodes[832], 'i'), (nodes[833], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 834. p290
    nodes[834] = addToKV(834, params['p290'])

    # 835. fused_reshape_add_add12
    # kernel 0
    output_size = 1572864
    nodes[835] = kaas.bufferSpec('835', output_size, const=False, ephemeral=True)
    arguments = [(nodes[835], 'o'), (nodes[833], 'i'), (nodes[834], 'i'), (nodes[826], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 836. fused_mean36
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a72', output_size, const=False, ephemeral=True))
    arguments = [(nodes[835], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[836] = kaas.bufferSpec('836', output_size, const=False, ephemeral=True)
    arguments = [(nodes[836], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 837. fused_subtract12
    # kernel 0
    output_size = 1572864
    nodes[837] = kaas.bufferSpec('837', output_size, const=False, ephemeral=True)
    arguments = [(nodes[837], 'o'), (nodes[835], 'i'), (nodes[836], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 838. fused_power_mean12
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a73', output_size, const=False, ephemeral=True))
    arguments = [(nodes[837], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[838] = kaas.bufferSpec('838', output_size, const=False, ephemeral=True)
    arguments = [(nodes[838], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 839. p291
    nodes[839] = addToKV(839, params['p291'])

    # 840. p292
    nodes[840] = addToKV(840, params['p292'])

    # 841. fused_add_sqrt_divide_multiply_add11
    # kernel 0
    output_size = 1572864
    nodes[841] = kaas.bufferSpec('841', output_size, const=False, ephemeral=True)
    arguments = [(nodes[841], 'o'), (nodes[838], 'i'), (nodes[837], 'i'), (nodes[839], 'i'), (nodes[840], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 842. reshape_nop
    nodes[842] = nodes[841]

    # 843. p293
    nodes[843] = addToKV(843, params['p293'])

    # 844. fused_nn_batch_matmul_311
    # kernel 0
    output_size = 1572864
    nodes[844] = kaas.bufferSpec('844', output_size, const=False, ephemeral=True)
    arguments = [(nodes[842], 'i'), (nodes[843], 'i'), (nodes[844], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 845. p294
    nodes[845] = addToKV(845, params['p294'])

    # 846. fused_reshape_add_reshape_transpose_reshape5
    # kernel 0
    output_size = 1572864
    nodes[846] = kaas.bufferSpec('846', output_size, const=False, ephemeral=True)
    arguments = [(nodes[846], 'o'), (nodes[844], 'i'), (nodes[845], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 847. p295
    nodes[847] = addToKV(847, params['p295'])

    # 848. fused_nn_batch_matmul_384
    # kernel 0
    output_size = 1572864
    nodes[848] = kaas.bufferSpec('848', output_size, const=False, ephemeral=True)
    arguments = [(nodes[842], 'i'), (nodes[847], 'i'), (nodes[848], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 849. p296
    nodes[849] = addToKV(849, params['p296'])

    # 850. fused_reshape_add_reshape_transpose_reshape_transpose18
    # kernel 0
    output_size = 1572864
    nodes[850] = kaas.bufferSpec('850', output_size, const=False, ephemeral=True)
    arguments = [(nodes[850], 'o'), (nodes[848], 'i'), (nodes[849], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 851. fused_nn_batch_matmul_55
    # kernel 0
    output_size = 9437184
    nodes[851] = kaas.bufferSpec('851', output_size, const=False, ephemeral=True)
    arguments = [(nodes[846], 'i'), (nodes[850], 'i'), (nodes[851], 'o'), ]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 852. fused_reshape_divide_add5
    # kernel 0
    output_size = 9437184
    nodes[852] = kaas.bufferSpec('852', output_size, const=False, ephemeral=True)
    arguments = [(nodes[852], 'o'), (nodes[851], 'i'), (nodes[23], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 853. fused_max18
    # kernel 0
    output_size = 24576
    nodes[853] = kaas.bufferSpec('853', output_size, const=False, ephemeral=True)
    arguments = [(nodes[852], 'i'), (nodes[853], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 854. fused_subtract_exp5
    # kernel 0
    output_size = 9437184
    nodes[854] = kaas.bufferSpec('854', output_size, const=False, ephemeral=True)
    arguments = [(nodes[854], 'o'), (nodes[852], 'i'), (nodes[853], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 855. fused_sum18
    # kernel 0
    output_size = 24576
    nodes[855] = kaas.bufferSpec('855', output_size, const=False, ephemeral=True)
    arguments = [(nodes[854], 'i'), (nodes[855], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 856. fused_divide_reshape5
    # kernel 0
    output_size = 9437184
    nodes[856] = kaas.bufferSpec('856', output_size, const=False, ephemeral=True)
    arguments = [(nodes[856], 'o'), (nodes[854], 'i'), (nodes[855], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 857. p297
    nodes[857] = addToKV(857, params['p297'])

    # 858. fused_nn_batch_matmul_385
    # kernel 0
    output_size = 1572864
    nodes[858] = kaas.bufferSpec('858', output_size, const=False, ephemeral=True)
    arguments = [(nodes[842], 'i'), (nodes[857], 'i'), (nodes[858], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 859. p298
    nodes[859] = addToKV(859, params['p298'])

    # 860. fused_reshape_add_reshape_transpose_reshape_transpose_118
    # kernel 0
    output_size = 1572864
    nodes[860] = kaas.bufferSpec('860', output_size, const=False, ephemeral=True)
    arguments = [(nodes[860], 'o'), (nodes[858], 'i'), (nodes[859], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 861. fused_nn_batch_matmul_45
    # kernel 0
    output_size = 1572864
    nodes[861] = kaas.bufferSpec('861', output_size, const=False, ephemeral=True)
    arguments = [(nodes[856], 'i'), (nodes[860], 'i'), (nodes[861], 'o'), ]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 862. fused_reshape_transpose_reshape5
    # kernel 0
    output_size = 1572864
    nodes[862] = kaas.bufferSpec('862', output_size, const=False, ephemeral=True)
    arguments = [(nodes[862], 'o'), (nodes[861], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 863. p299
    nodes[863] = addToKV(863, params['p299'])

    # 864. fused_nn_batch_matmul_310
    # kernel 0
    output_size = 1572864
    nodes[864] = kaas.bufferSpec('864', output_size, const=False, ephemeral=True)
    arguments = [(nodes[862], 'i'), (nodes[863], 'i'), (nodes[864], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 865. p300
    nodes[865] = addToKV(865, params['p300'])

    # 866. fused_reshape_add_add11
    # kernel 0
    output_size = 1572864
    nodes[866] = kaas.bufferSpec('866', output_size, const=False, ephemeral=True)
    arguments = [(nodes[866], 'o'), (nodes[864], 'i'), (nodes[865], 'i'), (nodes[841], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 867. fused_mean37
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a74', output_size, const=False, ephemeral=True))
    arguments = [(nodes[866], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[867] = kaas.bufferSpec('867', output_size, const=False, ephemeral=True)
    arguments = [(nodes[867], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 868. fused_subtract11
    # kernel 0
    output_size = 1572864
    nodes[868] = kaas.bufferSpec('868', output_size, const=False, ephemeral=True)
    arguments = [(nodes[868], 'o'), (nodes[866], 'i'), (nodes[867], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 869. fused_power_mean11
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a75', output_size, const=False, ephemeral=True))
    arguments = [(nodes[868], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[869] = kaas.bufferSpec('869', output_size, const=False, ephemeral=True)
    arguments = [(nodes[869], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 870. p301
    nodes[870] = addToKV(870, params['p301'])

    # 871. p302
    nodes[871] = addToKV(871, params['p302'])

    # 872. fused_add_sqrt_divide_multiply_add10
    # kernel 0
    output_size = 1572864
    nodes[872] = kaas.bufferSpec('872', output_size, const=False, ephemeral=True)
    arguments = [(nodes[872], 'o'), (nodes[869], 'i'), (nodes[868], 'i'), (nodes[870], 'i'), (nodes[871], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 873. reshape_nop
    nodes[873] = nodes[872]

    # 874. p303
    nodes[874] = addToKV(874, params['p303'])

    # 875. fused_nn_batch_matmul_25
    # kernel 0
    output_size = 6291456
    nodes[875] = kaas.bufferSpec('875', output_size, const=False, ephemeral=True)
    arguments = [(nodes[873], 'i'), (nodes[874], 'i'), (nodes[875], 'o'), ]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 876. p304
    nodes[876] = addToKV(876, params['p304'])

    # 877. fused_reshape_add_multiply_divide_erf_add_multiply_reshape5
    # kernel 0
    output_size = 6291456
    nodes[877] = kaas.bufferSpec('877', output_size, const=False, ephemeral=True)
    arguments = [(nodes[877], 'o'), (nodes[875], 'i'), (nodes[876], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 878. p305
    nodes[878] = addToKV(878, params['p305'])

    # 879. fused_nn_batch_matmul_15
    # kernel 0
    output_size = 1572864
    nodes[879] = kaas.bufferSpec('879', output_size, const=False, ephemeral=True)
    arguments = [(nodes[877], 'i'), (nodes[878], 'i'), (nodes[879], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 880. p306
    nodes[880] = addToKV(880, params['p306'])

    # 881. fused_reshape_add_add10
    # kernel 0
    output_size = 1572864
    nodes[881] = kaas.bufferSpec('881', output_size, const=False, ephemeral=True)
    arguments = [(nodes[881], 'o'), (nodes[879], 'i'), (nodes[880], 'i'), (nodes[872], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 882. fused_mean38
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a76', output_size, const=False, ephemeral=True))
    arguments = [(nodes[881], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[882] = kaas.bufferSpec('882', output_size, const=False, ephemeral=True)
    arguments = [(nodes[882], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 883. fused_subtract10
    # kernel 0
    output_size = 1572864
    nodes[883] = kaas.bufferSpec('883', output_size, const=False, ephemeral=True)
    arguments = [(nodes[883], 'o'), (nodes[881], 'i'), (nodes[882], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 884. fused_power_mean10
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a77', output_size, const=False, ephemeral=True))
    arguments = [(nodes[883], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[884] = kaas.bufferSpec('884', output_size, const=False, ephemeral=True)
    arguments = [(nodes[884], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 885. p307
    nodes[885] = addToKV(885, params['p307'])

    # 886. p308
    nodes[886] = addToKV(886, params['p308'])

    # 887. fused_add_sqrt_divide_multiply_add9
    # kernel 0
    output_size = 1572864
    nodes[887] = kaas.bufferSpec('887', output_size, const=False, ephemeral=True)
    arguments = [(nodes[887], 'o'), (nodes[884], 'i'), (nodes[883], 'i'), (nodes[885], 'i'), (nodes[886], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 888. reshape_nop
    nodes[888] = nodes[887]

    # 889. p309
    nodes[889] = addToKV(889, params['p309'])

    # 890. fused_nn_batch_matmul_39
    # kernel 0
    output_size = 1572864
    nodes[890] = kaas.bufferSpec('890', output_size, const=False, ephemeral=True)
    arguments = [(nodes[888], 'i'), (nodes[889], 'i'), (nodes[890], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 891. p310
    nodes[891] = addToKV(891, params['p310'])

    # 892. fused_reshape_add_reshape_transpose_reshape4
    # kernel 0
    output_size = 1572864
    nodes[892] = kaas.bufferSpec('892', output_size, const=False, ephemeral=True)
    arguments = [(nodes[892], 'o'), (nodes[890], 'i'), (nodes[891], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 893. p311
    nodes[893] = addToKV(893, params['p311'])

    # 894. fused_nn_batch_matmul_386
    # kernel 0
    output_size = 1572864
    nodes[894] = kaas.bufferSpec('894', output_size, const=False, ephemeral=True)
    arguments = [(nodes[888], 'i'), (nodes[893], 'i'), (nodes[894], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 895. p312
    nodes[895] = addToKV(895, params['p312'])

    # 896. fused_reshape_add_reshape_transpose_reshape_transpose19
    # kernel 0
    output_size = 1572864
    nodes[896] = kaas.bufferSpec('896', output_size, const=False, ephemeral=True)
    arguments = [(nodes[896], 'o'), (nodes[894], 'i'), (nodes[895], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 897. fused_nn_batch_matmul_54
    # kernel 0
    output_size = 9437184
    nodes[897] = kaas.bufferSpec('897', output_size, const=False, ephemeral=True)
    arguments = [(nodes[892], 'i'), (nodes[896], 'i'), (nodes[897], 'o'), ]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 898. fused_reshape_divide_add4
    # kernel 0
    output_size = 9437184
    nodes[898] = kaas.bufferSpec('898', output_size, const=False, ephemeral=True)
    arguments = [(nodes[898], 'o'), (nodes[897], 'i'), (nodes[23], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 899. fused_max19
    # kernel 0
    output_size = 24576
    nodes[899] = kaas.bufferSpec('899', output_size, const=False, ephemeral=True)
    arguments = [(nodes[898], 'i'), (nodes[899], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 900. fused_subtract_exp4
    # kernel 0
    output_size = 9437184
    nodes[900] = kaas.bufferSpec('900', output_size, const=False, ephemeral=True)
    arguments = [(nodes[900], 'o'), (nodes[898], 'i'), (nodes[899], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 901. fused_sum19
    # kernel 0
    output_size = 24576
    nodes[901] = kaas.bufferSpec('901', output_size, const=False, ephemeral=True)
    arguments = [(nodes[900], 'i'), (nodes[901], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 902. fused_divide_reshape4
    # kernel 0
    output_size = 9437184
    nodes[902] = kaas.bufferSpec('902', output_size, const=False, ephemeral=True)
    arguments = [(nodes[902], 'o'), (nodes[900], 'i'), (nodes[901], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 903. p313
    nodes[903] = addToKV(903, params['p313'])

    # 904. fused_nn_batch_matmul_387
    # kernel 0
    output_size = 1572864
    nodes[904] = kaas.bufferSpec('904', output_size, const=False, ephemeral=True)
    arguments = [(nodes[888], 'i'), (nodes[903], 'i'), (nodes[904], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 905. p314
    nodes[905] = addToKV(905, params['p314'])

    # 906. fused_reshape_add_reshape_transpose_reshape_transpose_119
    # kernel 0
    output_size = 1572864
    nodes[906] = kaas.bufferSpec('906', output_size, const=False, ephemeral=True)
    arguments = [(nodes[906], 'o'), (nodes[904], 'i'), (nodes[905], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 907. fused_nn_batch_matmul_44
    # kernel 0
    output_size = 1572864
    nodes[907] = kaas.bufferSpec('907', output_size, const=False, ephemeral=True)
    arguments = [(nodes[902], 'i'), (nodes[906], 'i'), (nodes[907], 'o'), ]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 908. fused_reshape_transpose_reshape4
    # kernel 0
    output_size = 1572864
    nodes[908] = kaas.bufferSpec('908', output_size, const=False, ephemeral=True)
    arguments = [(nodes[908], 'o'), (nodes[907], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 909. p315
    nodes[909] = addToKV(909, params['p315'])

    # 910. fused_nn_batch_matmul_38
    # kernel 0
    output_size = 1572864
    nodes[910] = kaas.bufferSpec('910', output_size, const=False, ephemeral=True)
    arguments = [(nodes[908], 'i'), (nodes[909], 'i'), (nodes[910], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 911. p316
    nodes[911] = addToKV(911, params['p316'])

    # 912. fused_reshape_add_add9
    # kernel 0
    output_size = 1572864
    nodes[912] = kaas.bufferSpec('912', output_size, const=False, ephemeral=True)
    arguments = [(nodes[912], 'o'), (nodes[910], 'i'), (nodes[911], 'i'), (nodes[887], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 913. fused_mean39
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a78', output_size, const=False, ephemeral=True))
    arguments = [(nodes[912], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[913] = kaas.bufferSpec('913', output_size, const=False, ephemeral=True)
    arguments = [(nodes[913], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 914. fused_subtract9
    # kernel 0
    output_size = 1572864
    nodes[914] = kaas.bufferSpec('914', output_size, const=False, ephemeral=True)
    arguments = [(nodes[914], 'o'), (nodes[912], 'i'), (nodes[913], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 915. fused_power_mean9
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a79', output_size, const=False, ephemeral=True))
    arguments = [(nodes[914], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[915] = kaas.bufferSpec('915', output_size, const=False, ephemeral=True)
    arguments = [(nodes[915], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 916. p317
    nodes[916] = addToKV(916, params['p317'])

    # 917. p318
    nodes[917] = addToKV(917, params['p318'])

    # 918. fused_add_sqrt_divide_multiply_add8
    # kernel 0
    output_size = 1572864
    nodes[918] = kaas.bufferSpec('918', output_size, const=False, ephemeral=True)
    arguments = [(nodes[918], 'o'), (nodes[915], 'i'), (nodes[914], 'i'), (nodes[916], 'i'), (nodes[917], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 919. reshape_nop
    nodes[919] = nodes[918]

    # 920. p319
    nodes[920] = addToKV(920, params['p319'])

    # 921. fused_nn_batch_matmul_24
    # kernel 0
    output_size = 6291456
    nodes[921] = kaas.bufferSpec('921', output_size, const=False, ephemeral=True)
    arguments = [(nodes[919], 'i'), (nodes[920], 'i'), (nodes[921], 'o'), ]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 922. p320
    nodes[922] = addToKV(922, params['p320'])

    # 923. fused_reshape_add_multiply_divide_erf_add_multiply_reshape4
    # kernel 0
    output_size = 6291456
    nodes[923] = kaas.bufferSpec('923', output_size, const=False, ephemeral=True)
    arguments = [(nodes[923], 'o'), (nodes[921], 'i'), (nodes[922], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 924. p321
    nodes[924] = addToKV(924, params['p321'])

    # 925. fused_nn_batch_matmul_14
    # kernel 0
    output_size = 1572864
    nodes[925] = kaas.bufferSpec('925', output_size, const=False, ephemeral=True)
    arguments = [(nodes[923], 'i'), (nodes[924], 'i'), (nodes[925], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 926. p322
    nodes[926] = addToKV(926, params['p322'])

    # 927. fused_reshape_add_add8
    # kernel 0
    output_size = 1572864
    nodes[927] = kaas.bufferSpec('927', output_size, const=False, ephemeral=True)
    arguments = [(nodes[927], 'o'), (nodes[925], 'i'), (nodes[926], 'i'), (nodes[918], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 928. fused_mean40
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a80', output_size, const=False, ephemeral=True))
    arguments = [(nodes[927], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[928] = kaas.bufferSpec('928', output_size, const=False, ephemeral=True)
    arguments = [(nodes[928], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 929. fused_subtract8
    # kernel 0
    output_size = 1572864
    nodes[929] = kaas.bufferSpec('929', output_size, const=False, ephemeral=True)
    arguments = [(nodes[929], 'o'), (nodes[927], 'i'), (nodes[928], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 930. fused_power_mean8
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a81', output_size, const=False, ephemeral=True))
    arguments = [(nodes[929], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[930] = kaas.bufferSpec('930', output_size, const=False, ephemeral=True)
    arguments = [(nodes[930], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 931. p323
    nodes[931] = addToKV(931, params['p323'])

    # 932. p324
    nodes[932] = addToKV(932, params['p324'])

    # 933. fused_add_sqrt_divide_multiply_add7
    # kernel 0
    output_size = 1572864
    nodes[933] = kaas.bufferSpec('933', output_size, const=False, ephemeral=True)
    arguments = [(nodes[933], 'o'), (nodes[930], 'i'), (nodes[929], 'i'), (nodes[931], 'i'), (nodes[932], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 934. reshape_nop
    nodes[934] = nodes[933]

    # 935. p325
    nodes[935] = addToKV(935, params['p325'])

    # 936. fused_nn_batch_matmul_37
    # kernel 0
    output_size = 1572864
    nodes[936] = kaas.bufferSpec('936', output_size, const=False, ephemeral=True)
    arguments = [(nodes[934], 'i'), (nodes[935], 'i'), (nodes[936], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 937. p326
    nodes[937] = addToKV(937, params['p326'])

    # 938. fused_reshape_add_reshape_transpose_reshape3
    # kernel 0
    output_size = 1572864
    nodes[938] = kaas.bufferSpec('938', output_size, const=False, ephemeral=True)
    arguments = [(nodes[938], 'o'), (nodes[936], 'i'), (nodes[937], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 939. p327
    nodes[939] = addToKV(939, params['p327'])

    # 940. fused_nn_batch_matmul_388
    # kernel 0
    output_size = 1572864
    nodes[940] = kaas.bufferSpec('940', output_size, const=False, ephemeral=True)
    arguments = [(nodes[934], 'i'), (nodes[939], 'i'), (nodes[940], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 941. p328
    nodes[941] = addToKV(941, params['p328'])

    # 942. fused_reshape_add_reshape_transpose_reshape_transpose20
    # kernel 0
    output_size = 1572864
    nodes[942] = kaas.bufferSpec('942', output_size, const=False, ephemeral=True)
    arguments = [(nodes[942], 'o'), (nodes[940], 'i'), (nodes[941], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 943. fused_nn_batch_matmul_53
    # kernel 0
    output_size = 9437184
    nodes[943] = kaas.bufferSpec('943', output_size, const=False, ephemeral=True)
    arguments = [(nodes[938], 'i'), (nodes[942], 'i'), (nodes[943], 'o'), ]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 944. fused_reshape_divide_add3
    # kernel 0
    output_size = 9437184
    nodes[944] = kaas.bufferSpec('944', output_size, const=False, ephemeral=True)
    arguments = [(nodes[944], 'o'), (nodes[943], 'i'), (nodes[23], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 945. fused_max20
    # kernel 0
    output_size = 24576
    nodes[945] = kaas.bufferSpec('945', output_size, const=False, ephemeral=True)
    arguments = [(nodes[944], 'i'), (nodes[945], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 946. fused_subtract_exp3
    # kernel 0
    output_size = 9437184
    nodes[946] = kaas.bufferSpec('946', output_size, const=False, ephemeral=True)
    arguments = [(nodes[946], 'o'), (nodes[944], 'i'), (nodes[945], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 947. fused_sum20
    # kernel 0
    output_size = 24576
    nodes[947] = kaas.bufferSpec('947', output_size, const=False, ephemeral=True)
    arguments = [(nodes[946], 'i'), (nodes[947], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 948. fused_divide_reshape3
    # kernel 0
    output_size = 9437184
    nodes[948] = kaas.bufferSpec('948', output_size, const=False, ephemeral=True)
    arguments = [(nodes[948], 'o'), (nodes[946], 'i'), (nodes[947], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 949. p329
    nodes[949] = addToKV(949, params['p329'])

    # 950. fused_nn_batch_matmul_389
    # kernel 0
    output_size = 1572864
    nodes[950] = kaas.bufferSpec('950', output_size, const=False, ephemeral=True)
    arguments = [(nodes[934], 'i'), (nodes[949], 'i'), (nodes[950], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 951. p330
    nodes[951] = addToKV(951, params['p330'])

    # 952. fused_reshape_add_reshape_transpose_reshape_transpose_120
    # kernel 0
    output_size = 1572864
    nodes[952] = kaas.bufferSpec('952', output_size, const=False, ephemeral=True)
    arguments = [(nodes[952], 'o'), (nodes[950], 'i'), (nodes[951], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 953. fused_nn_batch_matmul_43
    # kernel 0
    output_size = 1572864
    nodes[953] = kaas.bufferSpec('953', output_size, const=False, ephemeral=True)
    arguments = [(nodes[948], 'i'), (nodes[952], 'i'), (nodes[953], 'o'), ]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 954. fused_reshape_transpose_reshape3
    # kernel 0
    output_size = 1572864
    nodes[954] = kaas.bufferSpec('954', output_size, const=False, ephemeral=True)
    arguments = [(nodes[954], 'o'), (nodes[953], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 955. p331
    nodes[955] = addToKV(955, params['p331'])

    # 956. fused_nn_batch_matmul_36
    # kernel 0
    output_size = 1572864
    nodes[956] = kaas.bufferSpec('956', output_size, const=False, ephemeral=True)
    arguments = [(nodes[954], 'i'), (nodes[955], 'i'), (nodes[956], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 957. p332
    nodes[957] = addToKV(957, params['p332'])

    # 958. fused_reshape_add_add7
    # kernel 0
    output_size = 1572864
    nodes[958] = kaas.bufferSpec('958', output_size, const=False, ephemeral=True)
    arguments = [(nodes[958], 'o'), (nodes[956], 'i'), (nodes[957], 'i'), (nodes[933], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 959. fused_mean41
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a82', output_size, const=False, ephemeral=True))
    arguments = [(nodes[958], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[959] = kaas.bufferSpec('959', output_size, const=False, ephemeral=True)
    arguments = [(nodes[959], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 960. fused_subtract7
    # kernel 0
    output_size = 1572864
    nodes[960] = kaas.bufferSpec('960', output_size, const=False, ephemeral=True)
    arguments = [(nodes[960], 'o'), (nodes[958], 'i'), (nodes[959], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 961. fused_power_mean7
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a83', output_size, const=False, ephemeral=True))
    arguments = [(nodes[960], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[961] = kaas.bufferSpec('961', output_size, const=False, ephemeral=True)
    arguments = [(nodes[961], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 962. p333
    nodes[962] = addToKV(962, params['p333'])

    # 963. p334
    nodes[963] = addToKV(963, params['p334'])

    # 964. fused_add_sqrt_divide_multiply_add6
    # kernel 0
    output_size = 1572864
    nodes[964] = kaas.bufferSpec('964', output_size, const=False, ephemeral=True)
    arguments = [(nodes[964], 'o'), (nodes[961], 'i'), (nodes[960], 'i'), (nodes[962], 'i'), (nodes[963], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 965. reshape_nop
    nodes[965] = nodes[964]

    # 966. p335
    nodes[966] = addToKV(966, params['p335'])

    # 967. fused_nn_batch_matmul_23
    # kernel 0
    output_size = 6291456
    nodes[967] = kaas.bufferSpec('967', output_size, const=False, ephemeral=True)
    arguments = [(nodes[965], 'i'), (nodes[966], 'i'), (nodes[967], 'o'), ]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 968. p336
    nodes[968] = addToKV(968, params['p336'])

    # 969. fused_reshape_add_multiply_divide_erf_add_multiply_reshape3
    # kernel 0
    output_size = 6291456
    nodes[969] = kaas.bufferSpec('969', output_size, const=False, ephemeral=True)
    arguments = [(nodes[969], 'o'), (nodes[967], 'i'), (nodes[968], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 970. p337
    nodes[970] = addToKV(970, params['p337'])

    # 971. fused_nn_batch_matmul_13
    # kernel 0
    output_size = 1572864
    nodes[971] = kaas.bufferSpec('971', output_size, const=False, ephemeral=True)
    arguments = [(nodes[969], 'i'), (nodes[970], 'i'), (nodes[971], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 972. p338
    nodes[972] = addToKV(972, params['p338'])

    # 973. fused_reshape_add_add6
    # kernel 0
    output_size = 1572864
    nodes[973] = kaas.bufferSpec('973', output_size, const=False, ephemeral=True)
    arguments = [(nodes[973], 'o'), (nodes[971], 'i'), (nodes[972], 'i'), (nodes[964], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 974. fused_mean42
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a84', output_size, const=False, ephemeral=True))
    arguments = [(nodes[973], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[974] = kaas.bufferSpec('974', output_size, const=False, ephemeral=True)
    arguments = [(nodes[974], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 975. fused_subtract6
    # kernel 0
    output_size = 1572864
    nodes[975] = kaas.bufferSpec('975', output_size, const=False, ephemeral=True)
    arguments = [(nodes[975], 'o'), (nodes[973], 'i'), (nodes[974], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 976. fused_power_mean6
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a85', output_size, const=False, ephemeral=True))
    arguments = [(nodes[975], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[976] = kaas.bufferSpec('976', output_size, const=False, ephemeral=True)
    arguments = [(nodes[976], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 977. p339
    nodes[977] = addToKV(977, params['p339'])

    # 978. p340
    nodes[978] = addToKV(978, params['p340'])

    # 979. fused_add_sqrt_divide_multiply_add5
    # kernel 0
    output_size = 1572864
    nodes[979] = kaas.bufferSpec('979', output_size, const=False, ephemeral=True)
    arguments = [(nodes[979], 'o'), (nodes[976], 'i'), (nodes[975], 'i'), (nodes[977], 'i'), (nodes[978], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 980. reshape_nop
    nodes[980] = nodes[979]

    # 981. p341
    nodes[981] = addToKV(981, params['p341'])

    # 982. fused_nn_batch_matmul_35
    # kernel 0
    output_size = 1572864
    nodes[982] = kaas.bufferSpec('982', output_size, const=False, ephemeral=True)
    arguments = [(nodes[980], 'i'), (nodes[981], 'i'), (nodes[982], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 983. p342
    nodes[983] = addToKV(983, params['p342'])

    # 984. fused_reshape_add_reshape_transpose_reshape2
    # kernel 0
    output_size = 1572864
    nodes[984] = kaas.bufferSpec('984', output_size, const=False, ephemeral=True)
    arguments = [(nodes[984], 'o'), (nodes[982], 'i'), (nodes[983], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 985. p343
    nodes[985] = addToKV(985, params['p343'])

    # 986. fused_nn_batch_matmul_390
    # kernel 0
    output_size = 1572864
    nodes[986] = kaas.bufferSpec('986', output_size, const=False, ephemeral=True)
    arguments = [(nodes[980], 'i'), (nodes[985], 'i'), (nodes[986], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 987. p344
    nodes[987] = addToKV(987, params['p344'])

    # 988. fused_reshape_add_reshape_transpose_reshape_transpose21
    # kernel 0
    output_size = 1572864
    nodes[988] = kaas.bufferSpec('988', output_size, const=False, ephemeral=True)
    arguments = [(nodes[988], 'o'), (nodes[986], 'i'), (nodes[987], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 989. fused_nn_batch_matmul_52
    # kernel 0
    output_size = 9437184
    nodes[989] = kaas.bufferSpec('989', output_size, const=False, ephemeral=True)
    arguments = [(nodes[984], 'i'), (nodes[988], 'i'), (nodes[989], 'o'), ]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 990. fused_reshape_divide_add2
    # kernel 0
    output_size = 9437184
    nodes[990] = kaas.bufferSpec('990', output_size, const=False, ephemeral=True)
    arguments = [(nodes[990], 'o'), (nodes[989], 'i'), (nodes[23], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 991. fused_max21
    # kernel 0
    output_size = 24576
    nodes[991] = kaas.bufferSpec('991', output_size, const=False, ephemeral=True)
    arguments = [(nodes[990], 'i'), (nodes[991], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 992. fused_subtract_exp2
    # kernel 0
    output_size = 9437184
    nodes[992] = kaas.bufferSpec('992', output_size, const=False, ephemeral=True)
    arguments = [(nodes[992], 'o'), (nodes[990], 'i'), (nodes[991], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 993. fused_sum21
    # kernel 0
    output_size = 24576
    nodes[993] = kaas.bufferSpec('993', output_size, const=False, ephemeral=True)
    arguments = [(nodes[992], 'i'), (nodes[993], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 994. fused_divide_reshape2
    # kernel 0
    output_size = 9437184
    nodes[994] = kaas.bufferSpec('994', output_size, const=False, ephemeral=True)
    arguments = [(nodes[994], 'o'), (nodes[992], 'i'), (nodes[993], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 995. p345
    nodes[995] = addToKV(995, params['p345'])

    # 996. fused_nn_batch_matmul_391
    # kernel 0
    output_size = 1572864
    nodes[996] = kaas.bufferSpec('996', output_size, const=False, ephemeral=True)
    arguments = [(nodes[980], 'i'), (nodes[995], 'i'), (nodes[996], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 997. p346
    nodes[997] = addToKV(997, params['p346'])

    # 998. fused_reshape_add_reshape_transpose_reshape_transpose_121
    # kernel 0
    output_size = 1572864
    nodes[998] = kaas.bufferSpec('998', output_size, const=False, ephemeral=True)
    arguments = [(nodes[998], 'o'), (nodes[996], 'i'), (nodes[997], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 999. fused_nn_batch_matmul_42
    # kernel 0
    output_size = 1572864
    nodes[999] = kaas.bufferSpec('999', output_size, const=False, ephemeral=True)
    arguments = [(nodes[994], 'i'), (nodes[998], 'i'), (nodes[999], 'o'), ]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 1000. fused_reshape_transpose_reshape2
    # kernel 0
    output_size = 1572864
    nodes[1000] = kaas.bufferSpec('1000', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1000], 'o'), (nodes[999], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 1001. p347
    nodes[1001] = addToKV(1001, params['p347'])

    # 1002. fused_nn_batch_matmul_34
    # kernel 0
    output_size = 1572864
    nodes[1002] = kaas.bufferSpec('1002', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1000], 'i'), (nodes[1001], 'i'), (nodes[1002], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 1003. p348
    nodes[1003] = addToKV(1003, params['p348'])

    # 1004. fused_reshape_add_add5
    # kernel 0
    output_size = 1572864
    nodes[1004] = kaas.bufferSpec('1004', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1004], 'o'), (nodes[1002], 'i'), (nodes[1003], 'i'), (nodes[979], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 1005. fused_mean43
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a86', output_size, const=False, ephemeral=True))
    arguments = [(nodes[1004], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[1005] = kaas.bufferSpec('1005', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1005], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 1006. fused_subtract5
    # kernel 0
    output_size = 1572864
    nodes[1006] = kaas.bufferSpec('1006', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1006], 'o'), (nodes[1004], 'i'), (nodes[1005], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 1007. fused_power_mean5
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a87', output_size, const=False, ephemeral=True))
    arguments = [(nodes[1006], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[1007] = kaas.bufferSpec('1007', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1007], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 1008. p349
    nodes[1008] = addToKV(1008, params['p349'])

    # 1009. p350
    nodes[1009] = addToKV(1009, params['p350'])

    # 1010. fused_add_sqrt_divide_multiply_add4
    # kernel 0
    output_size = 1572864
    nodes[1010] = kaas.bufferSpec('1010', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1010], 'o'), (nodes[1007], 'i'), (nodes[1006], 'i'), (nodes[1008], 'i'), (nodes[1009], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 1011. reshape_nop
    nodes[1011] = nodes[1010]

    # 1012. p351
    nodes[1012] = addToKV(1012, params['p351'])

    # 1013. fused_nn_batch_matmul_22
    # kernel 0
    output_size = 6291456
    nodes[1013] = kaas.bufferSpec('1013', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1011], 'i'), (nodes[1012], 'i'), (nodes[1013], 'o'), ]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 1014. p352
    nodes[1014] = addToKV(1014, params['p352'])

    # 1015. fused_reshape_add_multiply_divide_erf_add_multiply_reshape2
    # kernel 0
    output_size = 6291456
    nodes[1015] = kaas.bufferSpec('1015', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1015], 'o'), (nodes[1013], 'i'), (nodes[1014], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 1016. p353
    nodes[1016] = addToKV(1016, params['p353'])

    # 1017. fused_nn_batch_matmul_12
    # kernel 0
    output_size = 1572864
    nodes[1017] = kaas.bufferSpec('1017', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1015], 'i'), (nodes[1016], 'i'), (nodes[1017], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 1018. p354
    nodes[1018] = addToKV(1018, params['p354'])

    # 1019. fused_reshape_add_add4
    # kernel 0
    output_size = 1572864
    nodes[1019] = kaas.bufferSpec('1019', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1019], 'o'), (nodes[1017], 'i'), (nodes[1018], 'i'), (nodes[1010], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 1020. fused_mean44
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a88', output_size, const=False, ephemeral=True))
    arguments = [(nodes[1019], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[1020] = kaas.bufferSpec('1020', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1020], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 1021. fused_subtract4
    # kernel 0
    output_size = 1572864
    nodes[1021] = kaas.bufferSpec('1021', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1021], 'o'), (nodes[1019], 'i'), (nodes[1020], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 1022. fused_power_mean4
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a89', output_size, const=False, ephemeral=True))
    arguments = [(nodes[1021], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[1022] = kaas.bufferSpec('1022', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1022], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 1023. p355
    nodes[1023] = addToKV(1023, params['p355'])

    # 1024. p356
    nodes[1024] = addToKV(1024, params['p356'])

    # 1025. fused_add_sqrt_divide_multiply_add3
    # kernel 0
    output_size = 1572864
    nodes[1025] = kaas.bufferSpec('1025', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1025], 'o'), (nodes[1022], 'i'), (nodes[1021], 'i'), (nodes[1023], 'i'), (nodes[1024], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 1026. reshape_nop
    nodes[1026] = nodes[1025]

    # 1027. p357
    nodes[1027] = addToKV(1027, params['p357'])

    # 1028. fused_nn_batch_matmul_33
    # kernel 0
    output_size = 1572864
    nodes[1028] = kaas.bufferSpec('1028', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1026], 'i'), (nodes[1027], 'i'), (nodes[1028], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 1029. p358
    nodes[1029] = addToKV(1029, params['p358'])

    # 1030. fused_reshape_add_reshape_transpose_reshape1
    # kernel 0
    output_size = 1572864
    nodes[1030] = kaas.bufferSpec('1030', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1030], 'o'), (nodes[1028], 'i'), (nodes[1029], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 1031. p359
    nodes[1031] = addToKV(1031, params['p359'])

    # 1032. fused_nn_batch_matmul_392
    # kernel 0
    output_size = 1572864
    nodes[1032] = kaas.bufferSpec('1032', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1026], 'i'), (nodes[1031], 'i'), (nodes[1032], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 1033. p360
    nodes[1033] = addToKV(1033, params['p360'])

    # 1034. fused_reshape_add_reshape_transpose_reshape_transpose22
    # kernel 0
    output_size = 1572864
    nodes[1034] = kaas.bufferSpec('1034', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1034], 'o'), (nodes[1032], 'i'), (nodes[1033], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 1035. fused_nn_batch_matmul_51
    # kernel 0
    output_size = 9437184
    nodes[1035] = kaas.bufferSpec('1035', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1030], 'i'), (nodes[1034], 'i'), (nodes[1035], 'o'), ]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 1036. fused_reshape_divide_add1
    # kernel 0
    output_size = 9437184
    nodes[1036] = kaas.bufferSpec('1036', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1036], 'o'), (nodes[1035], 'i'), (nodes[23], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 1037. fused_max22
    # kernel 0
    output_size = 24576
    nodes[1037] = kaas.bufferSpec('1037', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1036], 'i'), (nodes[1037], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 1038. fused_subtract_exp1
    # kernel 0
    output_size = 9437184
    nodes[1038] = kaas.bufferSpec('1038', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1038], 'o'), (nodes[1036], 'i'), (nodes[1037], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 1039. fused_sum22
    # kernel 0
    output_size = 24576
    nodes[1039] = kaas.bufferSpec('1039', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1038], 'i'), (nodes[1039], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 1040. fused_divide_reshape1
    # kernel 0
    output_size = 9437184
    nodes[1040] = kaas.bufferSpec('1040', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1040], 'o'), (nodes[1038], 'i'), (nodes[1039], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 1041. p361
    nodes[1041] = addToKV(1041, params['p361'])

    # 1042. fused_nn_batch_matmul_393
    # kernel 0
    output_size = 1572864
    nodes[1042] = kaas.bufferSpec('1042', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1026], 'i'), (nodes[1041], 'i'), (nodes[1042], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 1043. p362
    nodes[1043] = addToKV(1043, params['p362'])

    # 1044. fused_reshape_add_reshape_transpose_reshape_transpose_122
    # kernel 0
    output_size = 1572864
    nodes[1044] = kaas.bufferSpec('1044', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1044], 'o'), (nodes[1042], 'i'), (nodes[1043], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 1045. fused_nn_batch_matmul_41
    # kernel 0
    output_size = 1572864
    nodes[1045] = kaas.bufferSpec('1045', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1040], 'i'), (nodes[1044], 'i'), (nodes[1045], 'o'), ]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 1046. fused_reshape_transpose_reshape1
    # kernel 0
    output_size = 1572864
    nodes[1046] = kaas.bufferSpec('1046', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1046], 'o'), (nodes[1045], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 1047. p363
    nodes[1047] = addToKV(1047, params['p363'])

    # 1048. fused_nn_batch_matmul_32
    # kernel 0
    output_size = 1572864
    nodes[1048] = kaas.bufferSpec('1048', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1046], 'i'), (nodes[1047], 'i'), (nodes[1048], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 1049. p364
    nodes[1049] = addToKV(1049, params['p364'])

    # 1050. fused_reshape_add_add3
    # kernel 0
    output_size = 1572864
    nodes[1050] = kaas.bufferSpec('1050', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1050], 'o'), (nodes[1048], 'i'), (nodes[1049], 'i'), (nodes[1025], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 1051. fused_mean45
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a90', output_size, const=False, ephemeral=True))
    arguments = [(nodes[1050], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[1051] = kaas.bufferSpec('1051', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1051], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 1052. fused_subtract3
    # kernel 0
    output_size = 1572864
    nodes[1052] = kaas.bufferSpec('1052', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1052], 'o'), (nodes[1050], 'i'), (nodes[1051], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 1053. fused_power_mean3
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a91', output_size, const=False, ephemeral=True))
    arguments = [(nodes[1052], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[1053] = kaas.bufferSpec('1053', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1053], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 1054. p365
    nodes[1054] = addToKV(1054, params['p365'])

    # 1055. p366
    nodes[1055] = addToKV(1055, params['p366'])

    # 1056. fused_add_sqrt_divide_multiply_add2
    # kernel 0
    output_size = 1572864
    nodes[1056] = kaas.bufferSpec('1056', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1056], 'o'), (nodes[1053], 'i'), (nodes[1052], 'i'), (nodes[1054], 'i'), (nodes[1055], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 1057. reshape_nop
    nodes[1057] = nodes[1056]

    # 1058. p367
    nodes[1058] = addToKV(1058, params['p367'])

    # 1059. fused_nn_batch_matmul_21
    # kernel 0
    output_size = 6291456
    nodes[1059] = kaas.bufferSpec('1059', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1057], 'i'), (nodes[1058], 'i'), (nodes[1059], 'o'), ]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 1060. p368
    nodes[1060] = addToKV(1060, params['p368'])

    # 1061. fused_reshape_add_multiply_divide_erf_add_multiply_reshape1
    # kernel 0
    output_size = 6291456
    nodes[1061] = kaas.bufferSpec('1061', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1061], 'o'), (nodes[1059], 'i'), (nodes[1060], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 1062. p369
    nodes[1062] = addToKV(1062, params['p369'])

    # 1063. fused_nn_batch_matmul_11
    # kernel 0
    output_size = 1572864
    nodes[1063] = kaas.bufferSpec('1063', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1061], 'i'), (nodes[1062], 'i'), (nodes[1063], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 1064. p370
    nodes[1064] = addToKV(1064, params['p370'])

    # 1065. fused_reshape_add_add2
    # kernel 0
    output_size = 1572864
    nodes[1065] = kaas.bufferSpec('1065', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1065], 'o'), (nodes[1063], 'i'), (nodes[1064], 'i'), (nodes[1056], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 1066. fused_mean46
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a92', output_size, const=False, ephemeral=True))
    arguments = [(nodes[1065], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[1066] = kaas.bufferSpec('1066', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1066], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 1067. fused_subtract2
    # kernel 0
    output_size = 1572864
    nodes[1067] = kaas.bufferSpec('1067', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1067], 'o'), (nodes[1065], 'i'), (nodes[1066], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 1068. fused_power_mean2
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a93', output_size, const=False, ephemeral=True))
    arguments = [(nodes[1067], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[1068] = kaas.bufferSpec('1068', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1068], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 1069. p371
    nodes[1069] = addToKV(1069, params['p371'])

    # 1070. p372
    nodes[1070] = addToKV(1070, params['p372'])

    # 1071. fused_add_sqrt_divide_multiply_add1
    # kernel 0
    output_size = 1572864
    nodes[1071] = kaas.bufferSpec('1071', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1071], 'o'), (nodes[1068], 'i'), (nodes[1067], 'i'), (nodes[1069], 'i'), (nodes[1070], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 1072. reshape_nop
    nodes[1072] = nodes[1071]

    # 1073. p373
    nodes[1073] = addToKV(1073, params['p373'])

    # 1074. fused_nn_batch_matmul_31
    # kernel 0
    output_size = 1572864
    nodes[1074] = kaas.bufferSpec('1074', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1072], 'i'), (nodes[1073], 'i'), (nodes[1074], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 1075. p374
    nodes[1075] = addToKV(1075, params['p374'])

    # 1076. fused_reshape_add_reshape_transpose_reshape
    # kernel 0
    output_size = 1572864
    nodes[1076] = kaas.bufferSpec('1076', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1076], 'o'), (nodes[1074], 'i'), (nodes[1075], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 1077. p375
    nodes[1077] = addToKV(1077, params['p375'])

    # 1078. fused_nn_batch_matmul_394
    # kernel 0
    output_size = 1572864
    nodes[1078] = kaas.bufferSpec('1078', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1072], 'i'), (nodes[1077], 'i'), (nodes[1078], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 1079. p376
    nodes[1079] = addToKV(1079, params['p376'])

    # 1080. fused_reshape_add_reshape_transpose_reshape_transpose23
    # kernel 0
    output_size = 1572864
    nodes[1080] = kaas.bufferSpec('1080', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1080], 'o'), (nodes[1078], 'i'), (nodes[1079], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 1081. fused_nn_batch_matmul_5
    # kernel 0
    output_size = 9437184
    nodes[1081] = kaas.bufferSpec('1081', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1076], 'i'), (nodes[1080], 'i'), (nodes[1081], 'o'), ]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 1082. fused_reshape_divide_add
    # kernel 0
    output_size = 9437184
    nodes[1082] = kaas.bufferSpec('1082', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1082], 'o'), (nodes[1081], 'i'), (nodes[23], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 1083. fused_max23
    # kernel 0
    output_size = 24576
    nodes[1083] = kaas.bufferSpec('1083', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1082], 'i'), (nodes[1083], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 1084. fused_subtract_exp
    # kernel 0
    output_size = 9437184
    nodes[1084] = kaas.bufferSpec('1084', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1084], 'o'), (nodes[1082], 'i'), (nodes[1083], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 1085. fused_sum23
    # kernel 0
    output_size = 24576
    nodes[1085] = kaas.bufferSpec('1085', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1084], 'i'), (nodes[1085], 'o'), ]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 1086. fused_divide_reshape
    # kernel 0
    output_size = 9437184
    nodes[1086] = kaas.bufferSpec('1086', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1086], 'o'), (nodes[1084], 'i'), (nodes[1085], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 1087. p377
    nodes[1087] = addToKV(1087, params['p377'])

    # 1088. fused_nn_batch_matmul_395
    # kernel 0
    output_size = 1572864
    nodes[1088] = kaas.bufferSpec('1088', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1072], 'i'), (nodes[1087], 'i'), (nodes[1088], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 1089. p378
    nodes[1089] = addToKV(1089, params['p378'])

    # 1090. fused_reshape_add_reshape_transpose_reshape_transpose_123
    # kernel 0
    output_size = 1572864
    nodes[1090] = kaas.bufferSpec('1090', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1090], 'o'), (nodes[1088], 'i'), (nodes[1089], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 1091. fused_nn_batch_matmul_4
    # kernel 0
    output_size = 1572864
    nodes[1091] = kaas.bufferSpec('1091', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1086], 'i'), (nodes[1090], 'i'), (nodes[1091], 'o'), ]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 1092. fused_reshape_transpose_reshape
    # kernel 0
    output_size = 1572864
    nodes[1092] = kaas.bufferSpec('1092', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1092], 'o'), (nodes[1091], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 1093. p379
    nodes[1093] = addToKV(1093, params['p379'])

    # 1094. fused_nn_batch_matmul_3
    # kernel 0
    output_size = 1572864
    nodes[1094] = kaas.bufferSpec('1094', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1092], 'i'), (nodes[1093], 'i'), (nodes[1094], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 1095. p380
    nodes[1095] = addToKV(1095, params['p380'])

    # 1096. fused_reshape_add_add1
    # kernel 0
    output_size = 1572864
    nodes[1096] = kaas.bufferSpec('1096', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1096], 'o'), (nodes[1094], 'i'), (nodes[1095], 'i'), (nodes[1071], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 1097. fused_mean47
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a94', output_size, const=False, ephemeral=True))
    arguments = [(nodes[1096], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[1097] = kaas.bufferSpec('1097', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1097], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 1098. fused_subtract1
    # kernel 0
    output_size = 1572864
    nodes[1098] = kaas.bufferSpec('1098', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1098], 'o'), (nodes[1096], 'i'), (nodes[1097], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 1099. fused_power_mean1
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a95', output_size, const=False, ephemeral=True))
    arguments = [(nodes[1098], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[1099] = kaas.bufferSpec('1099', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1099], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 1100. p381
    nodes[1100] = addToKV(1100, params['p381'])

    # 1101. p382
    nodes[1101] = addToKV(1101, params['p382'])

    # 1102. fused_add_sqrt_divide_multiply_add
    # kernel 0
    output_size = 1572864
    nodes[1102] = kaas.bufferSpec('1102', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1102], 'o'), (nodes[1099], 'i'), (nodes[1098], 'i'), (nodes[1100], 'i'), (nodes[1101], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 1103. reshape_nop
    nodes[1103] = nodes[1102]

    # 1104. p383
    nodes[1104] = addToKV(1104, params['p383'])

    # 1105. fused_nn_batch_matmul_2
    # kernel 0
    output_size = 6291456
    nodes[1105] = kaas.bufferSpec('1105', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1103], 'i'), (nodes[1104], 'i'), (nodes[1105], 'o'), ]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 1106. p384
    nodes[1106] = addToKV(1106, params['p384'])

    # 1107. fused_reshape_add_multiply_divide_erf_add_multiply_reshape
    # kernel 0
    output_size = 6291456
    nodes[1107] = kaas.bufferSpec('1107', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1107], 'o'), (nodes[1105], 'i'), (nodes[1106], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 1108. p385
    nodes[1108] = addToKV(1108, params['p385'])

    # 1109. fused_nn_batch_matmul_1
    # kernel 0
    output_size = 1572864
    nodes[1109] = kaas.bufferSpec('1109', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1107], 'i'), (nodes[1108], 'i'), (nodes[1109], 'o'), ]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 1110. p386
    nodes[1110] = addToKV(1110, params['p386'])

    # 1111. fused_reshape_add_add
    # kernel 0
    output_size = 1572864
    nodes[1111] = kaas.bufferSpec('1111', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1111], 'o'), (nodes[1109], 'i'), (nodes[1110], 'i'), (nodes[1102], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 1112. fused_mean48
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a96', output_size, const=False, ephemeral=True))
    arguments = [(nodes[1111], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[1112] = kaas.bufferSpec('1112', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1112], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 1113. fused_subtract
    # kernel 0
    output_size = 1572864
    nodes[1113] = kaas.bufferSpec('1113', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1113], 'o'), (nodes[1111], 'i'), (nodes[1112], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 1114. fused_power_mean
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a97', output_size, const=False, ephemeral=True))
    arguments = [(nodes[1113], 'i'), (imm[0], 'o'), ]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[1114] = kaas.bufferSpec('1114', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1114], 'o'), (imm[0], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 1115. p387
    nodes[1115] = addToKV(1115, params['p387'])

    # 1116. p388
    nodes[1116] = addToKV(1116, params['p388'])

    # 1117. fused_add_sqrt_divide_multiply_add_reshape
    # kernel 0
    output_size = 1572864
    nodes[1117] = kaas.bufferSpec('1117', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1117], 'o'), (nodes[1113], 'i'), (nodes[1114], 'i'), (nodes[1115], 'i'), (nodes[1116], 'i'), ]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_reshape_kernel0', path, shapes, arguments))

    # 1118. p389
    nodes[1118] = addToKV(1118, params['p389'])

    # 1119. fused_nn_batch_matmul
    # kernel 0
    output_size = 3072
    nodes[1119] = kaas.bufferSpec('1119', output_size, const=False, ephemeral=True)
    arguments = [(nodes[1117], 'i'), (nodes[1118], 'i'), (nodes[1119], 'o'), ]
    shapes = [(1, 6, 1),  (2, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_kernel0', path, shapes, arguments))

    # 1120. p390
    nodes[1120] = addToKV(1120, params['p390'])

    # 1121. fused_reshape_add_split
    # kernel 0
    output_size = 1536
    nodes["1121_0"] = kaas.bufferSpec('1121_0', output_size, const=False, ephemeral=True)
    arguments = [(nodes['1121_0'], 'o'), (nodes[1119], 'i'), (nodes[1120], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_split_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes["1121_1"] = kaas.bufferSpec('1121_1', output_size, const=False, ephemeral=True)
    arguments = [(nodes['1121_1'], 'o'), (nodes[1119], 'i'), (nodes[1120], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_split_kernel1', path, shapes, arguments))

    # 1122. fused_squeeze
    # kernel 0
    output_size = 1536
    nodes[1122] = kaas.bufferSpec('1122', output_size, const=False, ephemeral=False)
    arguments = [(nodes[1122], 'o'), (nodes['1121_1'], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_squeeze_kernel0', path, shapes, arguments))

    # 1123. fused_squeeze_1
    # kernel 0
    output_size = 1536
    nodes[1123] = kaas.bufferSpec('1123', output_size, const=False, ephemeral=False)
    arguments = [(nodes[1123], 'o'), (nodes['1121_0'], 'i'), ]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_squeeze_1_kernel0', path, shapes, arguments))

    req = kaas.kaasReq(kerns)
    return req
