import pathlib
import pickle

import kaas
import numpy as np

redisPwd = "Cd+OBWBEAXV0o2fg5yDrMjD9JUkW7J6MATWuGlRtkQXk/CBvf2HYEjKDYw4FC+eWPeVR8cQKWr7IztZy"
testPath = pathlib.Path(__file__).resolve().parent



# Adds the given array to the kv with name node_num.
def addToKV(node_num, arr, const=True, ephemeral=False):
    nByte = arr.nbytes
    buff = kaas.bufferSpec(str(node_num), nByte, const=const, ephemeral=ephemeral)
    return buff


def loadParams(param_path):
    params = pickle.load(open(param_path, 'rb'))
    return params


def makeKern(name_func, path, shapes, arguments):
    return kaas.kernelSpec(path, name_func, shapes[0], shapes[1], arguments=arguments)


def createReq(params, cubinPath, mode='direct'):
    nodes = dict()
    kerns = []
    path = cubinPath
    inp = np.zeros((1, 384))
    nodes[0] = addToKV(0, inp, const=False, ephemeral=False)

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
    output_size = 9437184
    nodes[6] = kaas.bufferSpec('6', output_size, const=False, ephemeral=True)
    arguments = [(nodes[6], 't'), (nodes[3], 'i'), (nodes[0], 'i'), (nodes[4], 'i'), (nodes[5], 'i'), (nodes[2], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_less_add_where_take_add_less_add_where_take_add_kernel0', path, shapes, arguments))

    # 7. fused_mean
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a0', output_size, const=False, ephemeral=True))
    arguments = [(nodes[6], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes[7] = kaas.bufferSpec('7', output_size, const=False, ephemeral=True)
    arguments = [(nodes[7], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 8. fused_subtract48
    # kernel 0
    output_size = 9437184
    nodes[8] = kaas.bufferSpec('8', output_size, const=False, ephemeral=True)
    arguments = [(nodes[8], 't'), (nodes[6], 'i'), (nodes[7], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 9. fused_power_mean48
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a1', output_size, const=False, ephemeral=True))
    arguments = [(nodes[8], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[7], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 10. p3
    nodes[9] = addToKV(9, params['p3'])

    # 11. p4
    nodes[10] = addToKV(10, params['p4'])

    # 12. fused_add_sqrt_divide_multiply_add47
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[7], 'i'), (nodes[8], 'i'), (nodes[9], 'i'), (nodes[10], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 13. reshape_nop
    nodes[6] = nodes[6]

    # 14. p5
    nodes[11] = addToKV(11, params['p5'])

    # 15. fused_nn_batch_matmul_347
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[11], 'i'), (nodes[8], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 16. p6
    nodes[12] = addToKV(12, params['p6'])

    # 17. fused_reshape_add_reshape_transpose_reshape23
    # kernel 0
    output_size = 9437184
    nodes[13] = kaas.bufferSpec('13', output_size, const=False, ephemeral=True)
    arguments = [(nodes[13], 't'), (nodes[8], 'i'), (nodes[12], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 18. p7
    nodes[14] = addToKV(14, params['p7'])

    # 19. fused_nn_batch_matmul_348
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[14], 'i'), (nodes[8], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 20. p8
    nodes[15] = addToKV(15, params['p8'])

    # 21. fused_reshape_add_reshape_transpose_reshape_transpose
    # kernel 0
    output_size = 9437184
    nodes[16] = kaas.bufferSpec('16', output_size, const=False, ephemeral=True)
    arguments = [(nodes[16], 't'), (nodes[8], 'i'), (nodes[15], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 22. fused_nn_batch_matmul_523
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[16], 'i'), (nodes[8], 't')]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 23. fused_expand_dims_expand_dims_cast_subtract_multiply
    # kernel 0
    arguments = [(nodes[7], 't'), (nodes[1], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_expand_dims_expand_dims_cast_subtract_multiply_kernel0', path, shapes, arguments))

    # 24. fused_reshape_divide_add23
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[8], 'i'), (nodes[7], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 25. fused_max
    # kernel 0
    output_size = 24576
    nodes[17] = kaas.bufferSpec('17', output_size, const=False, ephemeral=True)
    arguments = [(nodes[16], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 26. fused_subtract_exp23
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[16], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 27. fused_sum
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 28. fused_divide_reshape23
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[8], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 29. p9
    nodes[18] = addToKV(18, params['p9'])

    # 30. fused_nn_batch_matmul_349
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[18], 'i'), (nodes[13], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 31. p10
    nodes[19] = addToKV(19, params['p10'])

    # 32. fused_reshape_add_reshape_transpose_reshape_transpose_1
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[13], 'i'), (nodes[19], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 33. fused_nn_batch_matmul_423
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[8], 'i'), (nodes[13], 't')]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 34. fused_reshape_transpose_reshape23
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[13], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 35. p11
    nodes[20] = addToKV(20, params['p11'])

    # 36. fused_nn_batch_matmul_346
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[20], 'i'), (nodes[13], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 37. p12
    nodes[21] = addToKV(21, params['p12'])

    # 38. fused_reshape_add_add47
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[13], 'i'), (nodes[21], 'i'), (nodes[6], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 39. fused_mean1
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a2', output_size, const=False, ephemeral=True))
    arguments = [(nodes[8], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 40. fused_subtract47
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[8], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 41. fused_power_mean47
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a3', output_size, const=False, ephemeral=True))
    arguments = [(nodes[13], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 42. p13
    nodes[22] = addToKV(22, params['p13'])

    # 43. p14
    nodes[23] = addToKV(23, params['p14'])

    # 44. fused_add_sqrt_divide_multiply_add46
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[17], 'i'), (nodes[13], 'i'), (nodes[22], 'i'), (nodes[23], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 45. reshape_nop
    nodes[6] = nodes[6]

    # 46. p15
    nodes[24] = addToKV(24, params['p15'])

    # 47. fused_nn_batch_matmul_223
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[24], 'i'), (nodes[16], 't')]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 48. p16
    nodes[25] = addToKV(25, params['p16'])

    # 49. fused_reshape_add_multiply_divide_erf_add_multiply_reshape23
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[16], 'i'), (nodes[25], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 50. p17
    nodes[26] = addToKV(26, params['p17'])

    # 51. fused_nn_batch_matmul_123
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[26], 'i'), (nodes[13], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 52. p18
    nodes[27] = addToKV(27, params['p18'])

    # 53. fused_reshape_add_add46
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[13], 'i'), (nodes[27], 'i'), (nodes[6], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 54. fused_mean2
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a4', output_size, const=False, ephemeral=True))
    arguments = [(nodes[16], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 55. fused_subtract46
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[16], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 56. fused_power_mean46
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a5', output_size, const=False, ephemeral=True))
    arguments = [(nodes[13], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 57. p19
    nodes[28] = addToKV(28, params['p19'])

    # 58. p20
    nodes[29] = addToKV(29, params['p20'])

    # 59. fused_add_sqrt_divide_multiply_add45
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[17], 'i'), (nodes[13], 'i'), (nodes[28], 'i'), (nodes[29], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 60. reshape_nop
    nodes[6] = nodes[6]

    # 61. p21
    nodes[30] = addToKV(30, params['p21'])

    # 62. fused_nn_batch_matmul_345
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[30], 'i'), (nodes[13], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 63. p22
    nodes[31] = addToKV(31, params['p22'])

    # 64. fused_reshape_add_reshape_transpose_reshape22
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[13], 'i'), (nodes[31], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 65. p23
    nodes[32] = addToKV(32, params['p23'])

    # 66. fused_nn_batch_matmul_350
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[32], 'i'), (nodes[13], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 67. p24
    nodes[33] = addToKV(33, params['p24'])

    # 68. fused_reshape_add_reshape_transpose_reshape_transpose1
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[13], 'i'), (nodes[33], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 69. fused_nn_batch_matmul_522
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[16], 'i'), (nodes[13], 't')]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 70. fused_reshape_divide_add22
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[13], 'i'), (nodes[7], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 71. fused_max1
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 72. fused_subtract_exp22
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[8], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 73. fused_sum1
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 74. fused_divide_reshape22
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[16], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 75. p25
    nodes[34] = addToKV(34, params['p25'])

    # 76. fused_nn_batch_matmul_351
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[34], 'i'), (nodes[8], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 77. p26
    nodes[35] = addToKV(35, params['p26'])

    # 78. fused_reshape_add_reshape_transpose_reshape_transpose_11
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[8], 'i'), (nodes[35], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 79. fused_nn_batch_matmul_422
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[16], 'i'), (nodes[8], 't')]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 80. fused_reshape_transpose_reshape22
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[8], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 81. p27
    nodes[36] = addToKV(36, params['p27'])

    # 82. fused_nn_batch_matmul_344
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[36], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 83. p28
    nodes[37] = addToKV(37, params['p28'])

    # 84. fused_reshape_add_add45
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[16], 'i'), (nodes[37], 'i'), (nodes[6], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 85. fused_mean3
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a6', output_size, const=False, ephemeral=True))
    arguments = [(nodes[8], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 86. fused_subtract45
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[8], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 87. fused_power_mean45
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a7', output_size, const=False, ephemeral=True))
    arguments = [(nodes[6], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 88. p29
    nodes[38] = addToKV(38, params['p29'])

    # 89. p30
    nodes[39] = addToKV(39, params['p30'])

    # 90. fused_add_sqrt_divide_multiply_add44
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[17], 'i'), (nodes[6], 'i'), (nodes[38], 'i'), (nodes[39], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 91. reshape_nop
    nodes[13] = nodes[13]

    # 92. p31
    nodes[40] = addToKV(40, params['p31'])

    # 93. fused_nn_batch_matmul_222
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[40], 'i'), (nodes[16], 't')]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 94. p32
    nodes[41] = addToKV(41, params['p32'])

    # 95. fused_reshape_add_multiply_divide_erf_add_multiply_reshape22
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[16], 'i'), (nodes[41], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 96. p33
    nodes[42] = addToKV(42, params['p33'])

    # 97. fused_nn_batch_matmul_122
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[42], 'i'), (nodes[6], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 98. p34
    nodes[43] = addToKV(43, params['p34'])

    # 99. fused_reshape_add_add44
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[6], 'i'), (nodes[43], 'i'), (nodes[13], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 100. fused_mean4
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a8', output_size, const=False, ephemeral=True))
    arguments = [(nodes[16], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 101. fused_subtract44
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[16], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 102. fused_power_mean44
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a9', output_size, const=False, ephemeral=True))
    arguments = [(nodes[6], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 103. p35
    nodes[44] = addToKV(44, params['p35'])

    # 104. p36
    nodes[45] = addToKV(45, params['p36'])

    # 105. fused_add_sqrt_divide_multiply_add43
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[17], 'i'), (nodes[6], 'i'), (nodes[44], 'i'), (nodes[45], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 106. reshape_nop
    nodes[8] = nodes[8]

    # 107. p37
    nodes[46] = addToKV(46, params['p37'])

    # 108. fused_nn_batch_matmul_343
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[46], 'i'), (nodes[6], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 109. p38
    nodes[47] = addToKV(47, params['p38'])

    # 110. fused_reshape_add_reshape_transpose_reshape21
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[6], 'i'), (nodes[47], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 111. p39
    nodes[48] = addToKV(48, params['p39'])

    # 112. fused_nn_batch_matmul_352
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[48], 'i'), (nodes[6], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 113. p40
    nodes[49] = addToKV(49, params['p40'])

    # 114. fused_reshape_add_reshape_transpose_reshape_transpose2
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[6], 'i'), (nodes[49], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 115. fused_nn_batch_matmul_521
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[16], 'i'), (nodes[6], 't')]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 116. fused_reshape_divide_add21
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[6], 'i'), (nodes[7], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 117. fused_max2
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 118. fused_subtract_exp21
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[13], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 119. fused_sum2
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 120. fused_divide_reshape21
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[16], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 121. p41
    nodes[50] = addToKV(50, params['p41'])

    # 122. fused_nn_batch_matmul_353
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[50], 'i'), (nodes[13], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 123. p42
    nodes[51] = addToKV(51, params['p42'])

    # 124. fused_reshape_add_reshape_transpose_reshape_transpose_12
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[13], 'i'), (nodes[51], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 125. fused_nn_batch_matmul_421
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[16], 'i'), (nodes[13], 't')]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 126. fused_reshape_transpose_reshape21
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[13], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 127. p43
    nodes[52] = addToKV(52, params['p43'])

    # 128. fused_nn_batch_matmul_342
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[52], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 129. p44
    nodes[53] = addToKV(53, params['p44'])

    # 130. fused_reshape_add_add43
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[16], 'i'), (nodes[53], 'i'), (nodes[8], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 131. fused_mean5
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a10', output_size, const=False, ephemeral=True))
    arguments = [(nodes[13], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 132. fused_subtract43
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[13], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 133. fused_power_mean43
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a11', output_size, const=False, ephemeral=True))
    arguments = [(nodes[6], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 134. p45
    nodes[54] = addToKV(54, params['p45'])

    # 135. p46
    nodes[55] = addToKV(55, params['p46'])

    # 136. fused_add_sqrt_divide_multiply_add42
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[17], 'i'), (nodes[6], 'i'), (nodes[54], 'i'), (nodes[55], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 137. reshape_nop
    nodes[16] = nodes[16]

    # 138. p47
    nodes[56] = addToKV(56, params['p47'])

    # 139. fused_nn_batch_matmul_221
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[56], 'i'), (nodes[8], 't')]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 140. p48
    nodes[57] = addToKV(57, params['p48'])

    # 141. fused_reshape_add_multiply_divide_erf_add_multiply_reshape21
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[8], 'i'), (nodes[57], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 142. p49
    nodes[58] = addToKV(58, params['p49'])

    # 143. fused_nn_batch_matmul_121
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[58], 'i'), (nodes[6], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 144. p50
    nodes[59] = addToKV(59, params['p50'])

    # 145. fused_reshape_add_add42
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[6], 'i'), (nodes[59], 'i'), (nodes[16], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 146. fused_mean6
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a12', output_size, const=False, ephemeral=True))
    arguments = [(nodes[8], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 147. fused_subtract42
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[8], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 148. fused_power_mean42
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a13', output_size, const=False, ephemeral=True))
    arguments = [(nodes[13], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 149. p51
    nodes[60] = addToKV(60, params['p51'])

    # 150. p52
    nodes[61] = addToKV(61, params['p52'])

    # 151. fused_add_sqrt_divide_multiply_add41
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[17], 'i'), (nodes[13], 'i'), (nodes[60], 'i'), (nodes[61], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 152. reshape_nop
    nodes[6] = nodes[6]

    # 153. p53
    nodes[62] = addToKV(62, params['p53'])

    # 154. fused_nn_batch_matmul_341
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[62], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 155. p54
    nodes[63] = addToKV(63, params['p54'])

    # 156. fused_reshape_add_reshape_transpose_reshape20
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[16], 'i'), (nodes[63], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 157. p55
    nodes[64] = addToKV(64, params['p55'])

    # 158. fused_nn_batch_matmul_354
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[64], 'i'), (nodes[13], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 159. p56
    nodes[65] = addToKV(65, params['p56'])

    # 160. fused_reshape_add_reshape_transpose_reshape_transpose3
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[13], 'i'), (nodes[65], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 161. fused_nn_batch_matmul_520
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[16], 'i'), (nodes[13], 't')]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 162. fused_reshape_divide_add20
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[13], 'i'), (nodes[7], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 163. fused_max3
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 164. fused_subtract_exp20
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[8], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 165. fused_sum3
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 166. fused_divide_reshape20
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[16], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 167. p57
    nodes[66] = addToKV(66, params['p57'])

    # 168. fused_nn_batch_matmul_355
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[66], 'i'), (nodes[8], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 169. p58
    nodes[67] = addToKV(67, params['p58'])

    # 170. fused_reshape_add_reshape_transpose_reshape_transpose_13
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[8], 'i'), (nodes[67], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 171. fused_nn_batch_matmul_420
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[16], 'i'), (nodes[8], 't')]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 172. fused_reshape_transpose_reshape20
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[8], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 173. p59
    nodes[68] = addToKV(68, params['p59'])

    # 174. fused_nn_batch_matmul_340
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[68], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 175. p60
    nodes[69] = addToKV(69, params['p60'])

    # 176. fused_reshape_add_add41
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[16], 'i'), (nodes[69], 'i'), (nodes[6], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 177. fused_mean7
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a14', output_size, const=False, ephemeral=True))
    arguments = [(nodes[8], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 178. fused_subtract41
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[8], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 179. fused_power_mean41
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a15', output_size, const=False, ephemeral=True))
    arguments = [(nodes[13], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 180. p61
    nodes[70] = addToKV(70, params['p61'])

    # 181. p62
    nodes[71] = addToKV(71, params['p62'])

    # 182. fused_add_sqrt_divide_multiply_add40
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[17], 'i'), (nodes[13], 'i'), (nodes[70], 'i'), (nodes[71], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 183. reshape_nop
    nodes[16] = nodes[16]

    # 184. p63
    nodes[72] = addToKV(72, params['p63'])

    # 185. fused_nn_batch_matmul_220
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[72], 'i'), (nodes[6], 't')]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 186. p64
    nodes[73] = addToKV(73, params['p64'])

    # 187. fused_reshape_add_multiply_divide_erf_add_multiply_reshape20
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[6], 'i'), (nodes[73], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 188. p65
    nodes[74] = addToKV(74, params['p65'])

    # 189. fused_nn_batch_matmul_120
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[74], 'i'), (nodes[13], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 190. p66
    nodes[75] = addToKV(75, params['p66'])

    # 191. fused_reshape_add_add40
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[13], 'i'), (nodes[75], 'i'), (nodes[16], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 192. fused_mean8
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a16', output_size, const=False, ephemeral=True))
    arguments = [(nodes[6], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 193. fused_subtract40
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[6], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 194. fused_power_mean40
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a17', output_size, const=False, ephemeral=True))
    arguments = [(nodes[8], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 195. p67
    nodes[76] = addToKV(76, params['p67'])

    # 196. p68
    nodes[77] = addToKV(77, params['p68'])

    # 197. fused_add_sqrt_divide_multiply_add39
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[17], 'i'), (nodes[8], 'i'), (nodes[76], 'i'), (nodes[77], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 198. reshape_nop
    nodes[13] = nodes[13]

    # 199. p69
    nodes[78] = addToKV(78, params['p69'])

    # 200. fused_nn_batch_matmul_339
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[78], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 201. p70
    nodes[79] = addToKV(79, params['p70'])

    # 202. fused_reshape_add_reshape_transpose_reshape19
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[16], 'i'), (nodes[79], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 203. p71
    nodes[80] = addToKV(80, params['p71'])

    # 204. fused_nn_batch_matmul_356
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[80], 'i'), (nodes[8], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 205. p72
    nodes[81] = addToKV(81, params['p72'])

    # 206. fused_reshape_add_reshape_transpose_reshape_transpose4
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[8], 'i'), (nodes[81], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 207. fused_nn_batch_matmul_519
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[16], 'i'), (nodes[8], 't')]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 208. fused_reshape_divide_add19
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[8], 'i'), (nodes[7], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 209. fused_max4
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 210. fused_subtract_exp19
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[6], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 211. fused_sum4
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 212. fused_divide_reshape19
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[16], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 213. p73
    nodes[82] = addToKV(82, params['p73'])

    # 214. fused_nn_batch_matmul_357
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[82], 'i'), (nodes[6], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 215. p74
    nodes[83] = addToKV(83, params['p74'])

    # 216. fused_reshape_add_reshape_transpose_reshape_transpose_14
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[6], 'i'), (nodes[83], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 217. fused_nn_batch_matmul_419
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[16], 'i'), (nodes[6], 't')]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 218. fused_reshape_transpose_reshape19
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[6], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 219. p75
    nodes[84] = addToKV(84, params['p75'])

    # 220. fused_nn_batch_matmul_338
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[84], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 221. p76
    nodes[85] = addToKV(85, params['p76'])

    # 222. fused_reshape_add_add39
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[16], 'i'), (nodes[85], 'i'), (nodes[13], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 223. fused_mean9
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a18', output_size, const=False, ephemeral=True))
    arguments = [(nodes[6], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 224. fused_subtract39
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[6], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 225. fused_power_mean39
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a19', output_size, const=False, ephemeral=True))
    arguments = [(nodes[8], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 226. p77
    nodes[86] = addToKV(86, params['p77'])

    # 227. p78
    nodes[87] = addToKV(87, params['p78'])

    # 228. fused_add_sqrt_divide_multiply_add38
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[17], 'i'), (nodes[8], 'i'), (nodes[86], 'i'), (nodes[87], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 229. reshape_nop
    nodes[16] = nodes[16]

    # 230. p79
    nodes[88] = addToKV(88, params['p79'])

    # 231. fused_nn_batch_matmul_219
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[88], 'i'), (nodes[13], 't')]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 232. p80
    nodes[89] = addToKV(89, params['p80'])

    # 233. fused_reshape_add_multiply_divide_erf_add_multiply_reshape19
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[13], 'i'), (nodes[89], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 234. p81
    nodes[90] = addToKV(90, params['p81'])

    # 235. fused_nn_batch_matmul_119
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[90], 'i'), (nodes[8], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 236. p82
    nodes[91] = addToKV(91, params['p82'])

    # 237. fused_reshape_add_add38
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[8], 'i'), (nodes[91], 'i'), (nodes[16], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 238. fused_mean10
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a20', output_size, const=False, ephemeral=True))
    arguments = [(nodes[13], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 239. fused_subtract38
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[13], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 240. fused_power_mean38
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a21', output_size, const=False, ephemeral=True))
    arguments = [(nodes[6], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 241. p83
    nodes[92] = addToKV(92, params['p83'])

    # 242. p84
    nodes[93] = addToKV(93, params['p84'])

    # 243. fused_add_sqrt_divide_multiply_add37
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[17], 'i'), (nodes[6], 'i'), (nodes[92], 'i'), (nodes[93], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 244. reshape_nop
    nodes[8] = nodes[8]

    # 245. p85
    nodes[94] = addToKV(94, params['p85'])

    # 246. fused_nn_batch_matmul_337
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[94], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 247. p86
    nodes[95] = addToKV(95, params['p86'])

    # 248. fused_reshape_add_reshape_transpose_reshape18
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[16], 'i'), (nodes[95], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 249. p87
    nodes[96] = addToKV(96, params['p87'])

    # 250. fused_nn_batch_matmul_358
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[96], 'i'), (nodes[6], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 251. p88
    nodes[97] = addToKV(97, params['p88'])

    # 252. fused_reshape_add_reshape_transpose_reshape_transpose5
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[6], 'i'), (nodes[97], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 253. fused_nn_batch_matmul_518
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[16], 'i'), (nodes[6], 't')]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 254. fused_reshape_divide_add18
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[6], 'i'), (nodes[7], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 255. fused_max5
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 256. fused_subtract_exp18
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[13], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 257. fused_sum5
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 258. fused_divide_reshape18
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[16], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 259. p89
    nodes[98] = addToKV(98, params['p89'])

    # 260. fused_nn_batch_matmul_359
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[98], 'i'), (nodes[13], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 261. p90
    nodes[99] = addToKV(99, params['p90'])

    # 262. fused_reshape_add_reshape_transpose_reshape_transpose_15
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[13], 'i'), (nodes[99], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 263. fused_nn_batch_matmul_418
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[16], 'i'), (nodes[13], 't')]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 264. fused_reshape_transpose_reshape18
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[13], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 265. p91
    nodes[100] = addToKV(100, params['p91'])

    # 266. fused_nn_batch_matmul_336
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[100], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 267. p92
    nodes[101] = addToKV(101, params['p92'])

    # 268. fused_reshape_add_add37
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[16], 'i'), (nodes[101], 'i'), (nodes[8], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 269. fused_mean11
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a22', output_size, const=False, ephemeral=True))
    arguments = [(nodes[13], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 270. fused_subtract37
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[13], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 271. fused_power_mean37
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a23', output_size, const=False, ephemeral=True))
    arguments = [(nodes[6], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 272. p93
    nodes[102] = addToKV(102, params['p93'])

    # 273. p94
    nodes[103] = addToKV(103, params['p94'])

    # 274. fused_add_sqrt_divide_multiply_add36
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[17], 'i'), (nodes[6], 'i'), (nodes[102], 'i'), (nodes[103], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 275. reshape_nop
    nodes[16] = nodes[16]

    # 276. p95
    nodes[104] = addToKV(104, params['p95'])

    # 277. fused_nn_batch_matmul_218
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[104], 'i'), (nodes[8], 't')]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 278. p96
    nodes[105] = addToKV(105, params['p96'])

    # 279. fused_reshape_add_multiply_divide_erf_add_multiply_reshape18
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[8], 'i'), (nodes[105], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 280. p97
    nodes[106] = addToKV(106, params['p97'])

    # 281. fused_nn_batch_matmul_118
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[106], 'i'), (nodes[6], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 282. p98
    nodes[107] = addToKV(107, params['p98'])

    # 283. fused_reshape_add_add36
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[6], 'i'), (nodes[107], 'i'), (nodes[16], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 284. fused_mean12
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a24', output_size, const=False, ephemeral=True))
    arguments = [(nodes[8], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 285. fused_subtract36
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[8], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 286. fused_power_mean36
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a25', output_size, const=False, ephemeral=True))
    arguments = [(nodes[13], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 287. p99
    nodes[108] = addToKV(108, params['p99'])

    # 288. p100
    nodes[109] = addToKV(109, params['p100'])

    # 289. fused_add_sqrt_divide_multiply_add35
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[17], 'i'), (nodes[13], 'i'), (nodes[108], 'i'), (nodes[109], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 290. reshape_nop
    nodes[6] = nodes[6]

    # 291. p101
    nodes[110] = addToKV(110, params['p101'])

    # 292. fused_nn_batch_matmul_335
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[110], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 293. p102
    nodes[111] = addToKV(111, params['p102'])

    # 294. fused_reshape_add_reshape_transpose_reshape17
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[16], 'i'), (nodes[111], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 295. p103
    nodes[112] = addToKV(112, params['p103'])

    # 296. fused_nn_batch_matmul_360
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[112], 'i'), (nodes[13], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 297. p104
    nodes[113] = addToKV(113, params['p104'])

    # 298. fused_reshape_add_reshape_transpose_reshape_transpose6
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[13], 'i'), (nodes[113], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 299. fused_nn_batch_matmul_517
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[16], 'i'), (nodes[13], 't')]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 300. fused_reshape_divide_add17
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[13], 'i'), (nodes[7], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 301. fused_max6
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 302. fused_subtract_exp17
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[8], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 303. fused_sum6
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 304. fused_divide_reshape17
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[16], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 305. p105
    nodes[114] = addToKV(114, params['p105'])

    # 306. fused_nn_batch_matmul_361
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[114], 'i'), (nodes[8], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 307. p106
    nodes[115] = addToKV(115, params['p106'])

    # 308. fused_reshape_add_reshape_transpose_reshape_transpose_16
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[8], 'i'), (nodes[115], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 309. fused_nn_batch_matmul_417
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[16], 'i'), (nodes[8], 't')]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 310. fused_reshape_transpose_reshape17
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[8], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 311. p107
    nodes[116] = addToKV(116, params['p107'])

    # 312. fused_nn_batch_matmul_334
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[116], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 313. p108
    nodes[117] = addToKV(117, params['p108'])

    # 314. fused_reshape_add_add35
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[16], 'i'), (nodes[117], 'i'), (nodes[6], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 315. fused_mean13
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a26', output_size, const=False, ephemeral=True))
    arguments = [(nodes[8], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 316. fused_subtract35
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[8], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 317. fused_power_mean35
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a27', output_size, const=False, ephemeral=True))
    arguments = [(nodes[13], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 318. p109
    nodes[118] = addToKV(118, params['p109'])

    # 319. p110
    nodes[119] = addToKV(119, params['p110'])

    # 320. fused_add_sqrt_divide_multiply_add34
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[17], 'i'), (nodes[13], 'i'), (nodes[118], 'i'), (nodes[119], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 321. reshape_nop
    nodes[16] = nodes[16]

    # 322. p111
    nodes[120] = addToKV(120, params['p111'])

    # 323. fused_nn_batch_matmul_217
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[120], 'i'), (nodes[6], 't')]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 324. p112
    nodes[121] = addToKV(121, params['p112'])

    # 325. fused_reshape_add_multiply_divide_erf_add_multiply_reshape17
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[6], 'i'), (nodes[121], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 326. p113
    nodes[122] = addToKV(122, params['p113'])

    # 327. fused_nn_batch_matmul_117
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[122], 'i'), (nodes[13], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 328. p114
    nodes[123] = addToKV(123, params['p114'])

    # 329. fused_reshape_add_add34
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[13], 'i'), (nodes[123], 'i'), (nodes[16], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 330. fused_mean14
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a28', output_size, const=False, ephemeral=True))
    arguments = [(nodes[6], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 331. fused_subtract34
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[6], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 332. fused_power_mean34
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a29', output_size, const=False, ephemeral=True))
    arguments = [(nodes[8], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 333. p115
    nodes[124] = addToKV(124, params['p115'])

    # 334. p116
    nodes[125] = addToKV(125, params['p116'])

    # 335. fused_add_sqrt_divide_multiply_add33
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[17], 'i'), (nodes[8], 'i'), (nodes[124], 'i'), (nodes[125], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 336. reshape_nop
    nodes[13] = nodes[13]

    # 337. p117
    nodes[126] = addToKV(126, params['p117'])

    # 338. fused_nn_batch_matmul_333
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[126], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 339. p118
    nodes[127] = addToKV(127, params['p118'])

    # 340. fused_reshape_add_reshape_transpose_reshape16
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[16], 'i'), (nodes[127], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 341. p119
    nodes[128] = addToKV(128, params['p119'])

    # 342. fused_nn_batch_matmul_362
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[128], 'i'), (nodes[8], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 343. p120
    nodes[129] = addToKV(129, params['p120'])

    # 344. fused_reshape_add_reshape_transpose_reshape_transpose7
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[8], 'i'), (nodes[129], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 345. fused_nn_batch_matmul_516
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[16], 'i'), (nodes[8], 't')]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 346. fused_reshape_divide_add16
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[8], 'i'), (nodes[7], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 347. fused_max7
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 348. fused_subtract_exp16
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[6], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 349. fused_sum7
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 350. fused_divide_reshape16
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[16], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 351. p121
    nodes[130] = addToKV(130, params['p121'])

    # 352. fused_nn_batch_matmul_363
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[130], 'i'), (nodes[6], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 353. p122
    nodes[131] = addToKV(131, params['p122'])

    # 354. fused_reshape_add_reshape_transpose_reshape_transpose_17
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[6], 'i'), (nodes[131], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 355. fused_nn_batch_matmul_416
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[16], 'i'), (nodes[6], 't')]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 356. fused_reshape_transpose_reshape16
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[6], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 357. p123
    nodes[132] = addToKV(132, params['p123'])

    # 358. fused_nn_batch_matmul_332
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[132], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 359. p124
    nodes[133] = addToKV(133, params['p124'])

    # 360. fused_reshape_add_add33
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[16], 'i'), (nodes[133], 'i'), (nodes[13], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 361. fused_mean15
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a30', output_size, const=False, ephemeral=True))
    arguments = [(nodes[6], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 362. fused_subtract33
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[6], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 363. fused_power_mean33
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a31', output_size, const=False, ephemeral=True))
    arguments = [(nodes[8], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 364. p125
    nodes[134] = addToKV(134, params['p125'])

    # 365. p126
    nodes[135] = addToKV(135, params['p126'])

    # 366. fused_add_sqrt_divide_multiply_add32
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[17], 'i'), (nodes[8], 'i'), (nodes[134], 'i'), (nodes[135], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 367. reshape_nop
    nodes[16] = nodes[16]

    # 368. p127
    nodes[136] = addToKV(136, params['p127'])

    # 369. fused_nn_batch_matmul_216
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[136], 'i'), (nodes[13], 't')]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 370. p128
    nodes[137] = addToKV(137, params['p128'])

    # 371. fused_reshape_add_multiply_divide_erf_add_multiply_reshape16
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[13], 'i'), (nodes[137], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 372. p129
    nodes[138] = addToKV(138, params['p129'])

    # 373. fused_nn_batch_matmul_116
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[138], 'i'), (nodes[8], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 374. p130
    nodes[139] = addToKV(139, params['p130'])

    # 375. fused_reshape_add_add32
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[8], 'i'), (nodes[139], 'i'), (nodes[16], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 376. fused_mean16
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a32', output_size, const=False, ephemeral=True))
    arguments = [(nodes[13], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 377. fused_subtract32
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[13], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 378. fused_power_mean32
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a33', output_size, const=False, ephemeral=True))
    arguments = [(nodes[6], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 379. p131
    nodes[140] = addToKV(140, params['p131'])

    # 380. p132
    nodes[141] = addToKV(141, params['p132'])

    # 381. fused_add_sqrt_divide_multiply_add31
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[17], 'i'), (nodes[6], 'i'), (nodes[140], 'i'), (nodes[141], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 382. reshape_nop
    nodes[8] = nodes[8]

    # 383. p133
    nodes[142] = addToKV(142, params['p133'])

    # 384. fused_nn_batch_matmul_331
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[142], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 385. p134
    nodes[143] = addToKV(143, params['p134'])

    # 386. fused_reshape_add_reshape_transpose_reshape15
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[16], 'i'), (nodes[143], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 387. p135
    nodes[144] = addToKV(144, params['p135'])

    # 388. fused_nn_batch_matmul_364
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[144], 'i'), (nodes[6], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 389. p136
    nodes[145] = addToKV(145, params['p136'])

    # 390. fused_reshape_add_reshape_transpose_reshape_transpose8
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[6], 'i'), (nodes[145], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 391. fused_nn_batch_matmul_515
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[16], 'i'), (nodes[6], 't')]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 392. fused_reshape_divide_add15
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[6], 'i'), (nodes[7], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 393. fused_max8
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 394. fused_subtract_exp15
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[13], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 395. fused_sum8
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 396. fused_divide_reshape15
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[16], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 397. p137
    nodes[146] = addToKV(146, params['p137'])

    # 398. fused_nn_batch_matmul_365
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[146], 'i'), (nodes[13], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 399. p138
    nodes[147] = addToKV(147, params['p138'])

    # 400. fused_reshape_add_reshape_transpose_reshape_transpose_18
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[13], 'i'), (nodes[147], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 401. fused_nn_batch_matmul_415
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[16], 'i'), (nodes[13], 't')]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 402. fused_reshape_transpose_reshape15
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[13], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 403. p139
    nodes[148] = addToKV(148, params['p139'])

    # 404. fused_nn_batch_matmul_330
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[148], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 405. p140
    nodes[149] = addToKV(149, params['p140'])

    # 406. fused_reshape_add_add31
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[16], 'i'), (nodes[149], 'i'), (nodes[8], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 407. fused_mean17
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a34', output_size, const=False, ephemeral=True))
    arguments = [(nodes[13], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 408. fused_subtract31
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[13], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 409. fused_power_mean31
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a35', output_size, const=False, ephemeral=True))
    arguments = [(nodes[6], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 410. p141
    nodes[150] = addToKV(150, params['p141'])

    # 411. p142
    nodes[151] = addToKV(151, params['p142'])

    # 412. fused_add_sqrt_divide_multiply_add30
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[17], 'i'), (nodes[6], 'i'), (nodes[150], 'i'), (nodes[151], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 413. reshape_nop
    nodes[16] = nodes[16]

    # 414. p143
    nodes[152] = addToKV(152, params['p143'])

    # 415. fused_nn_batch_matmul_215
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[152], 'i'), (nodes[8], 't')]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 416. p144
    nodes[153] = addToKV(153, params['p144'])

    # 417. fused_reshape_add_multiply_divide_erf_add_multiply_reshape15
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[8], 'i'), (nodes[153], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 418. p145
    nodes[154] = addToKV(154, params['p145'])

    # 419. fused_nn_batch_matmul_115
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[154], 'i'), (nodes[6], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 420. p146
    nodes[155] = addToKV(155, params['p146'])

    # 421. fused_reshape_add_add30
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[6], 'i'), (nodes[155], 'i'), (nodes[16], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 422. fused_mean18
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a36', output_size, const=False, ephemeral=True))
    arguments = [(nodes[8], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 423. fused_subtract30
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[8], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 424. fused_power_mean30
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a37', output_size, const=False, ephemeral=True))
    arguments = [(nodes[13], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 425. p147
    nodes[156] = addToKV(156, params['p147'])

    # 426. p148
    nodes[157] = addToKV(157, params['p148'])

    # 427. fused_add_sqrt_divide_multiply_add29
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[17], 'i'), (nodes[13], 'i'), (nodes[156], 'i'), (nodes[157], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 428. reshape_nop
    nodes[6] = nodes[6]

    # 429. p149
    nodes[158] = addToKV(158, params['p149'])

    # 430. fused_nn_batch_matmul_329
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[158], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 431. p150
    nodes[159] = addToKV(159, params['p150'])

    # 432. fused_reshape_add_reshape_transpose_reshape14
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[16], 'i'), (nodes[159], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 433. p151
    nodes[160] = addToKV(160, params['p151'])

    # 434. fused_nn_batch_matmul_366
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[160], 'i'), (nodes[13], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 435. p152
    nodes[161] = addToKV(161, params['p152'])

    # 436. fused_reshape_add_reshape_transpose_reshape_transpose9
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[13], 'i'), (nodes[161], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 437. fused_nn_batch_matmul_514
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[16], 'i'), (nodes[13], 't')]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 438. fused_reshape_divide_add14
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[13], 'i'), (nodes[7], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 439. fused_max9
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 440. fused_subtract_exp14
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[8], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 441. fused_sum9
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 442. fused_divide_reshape14
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[16], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 443. p153
    nodes[162] = addToKV(162, params['p153'])

    # 444. fused_nn_batch_matmul_367
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[162], 'i'), (nodes[8], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 445. p154
    nodes[163] = addToKV(163, params['p154'])

    # 446. fused_reshape_add_reshape_transpose_reshape_transpose_19
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[8], 'i'), (nodes[163], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 447. fused_nn_batch_matmul_414
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[16], 'i'), (nodes[8], 't')]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 448. fused_reshape_transpose_reshape14
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[8], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 449. p155
    nodes[164] = addToKV(164, params['p155'])

    # 450. fused_nn_batch_matmul_328
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[164], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 451. p156
    nodes[165] = addToKV(165, params['p156'])

    # 452. fused_reshape_add_add29
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[16], 'i'), (nodes[165], 'i'), (nodes[6], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 453. fused_mean19
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a38', output_size, const=False, ephemeral=True))
    arguments = [(nodes[8], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 454. fused_subtract29
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[8], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 455. fused_power_mean29
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a39', output_size, const=False, ephemeral=True))
    arguments = [(nodes[13], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 456. p157
    nodes[166] = addToKV(166, params['p157'])

    # 457. p158
    nodes[167] = addToKV(167, params['p158'])

    # 458. fused_add_sqrt_divide_multiply_add28
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[17], 'i'), (nodes[13], 'i'), (nodes[166], 'i'), (nodes[167], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 459. reshape_nop
    nodes[16] = nodes[16]

    # 460. p159
    nodes[168] = addToKV(168, params['p159'])

    # 461. fused_nn_batch_matmul_214
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[168], 'i'), (nodes[6], 't')]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 462. p160
    nodes[169] = addToKV(169, params['p160'])

    # 463. fused_reshape_add_multiply_divide_erf_add_multiply_reshape14
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[6], 'i'), (nodes[169], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 464. p161
    nodes[170] = addToKV(170, params['p161'])

    # 465. fused_nn_batch_matmul_114
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[170], 'i'), (nodes[13], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 466. p162
    nodes[171] = addToKV(171, params['p162'])

    # 467. fused_reshape_add_add28
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[13], 'i'), (nodes[171], 'i'), (nodes[16], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 468. fused_mean20
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a40', output_size, const=False, ephemeral=True))
    arguments = [(nodes[6], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 469. fused_subtract28
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[6], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 470. fused_power_mean28
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a41', output_size, const=False, ephemeral=True))
    arguments = [(nodes[8], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 471. p163
    nodes[172] = addToKV(172, params['p163'])

    # 472. p164
    nodes[173] = addToKV(173, params['p164'])

    # 473. fused_add_sqrt_divide_multiply_add27
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[17], 'i'), (nodes[8], 'i'), (nodes[172], 'i'), (nodes[173], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 474. reshape_nop
    nodes[13] = nodes[13]

    # 475. p165
    nodes[174] = addToKV(174, params['p165'])

    # 476. fused_nn_batch_matmul_327
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[174], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 477. p166
    nodes[175] = addToKV(175, params['p166'])

    # 478. fused_reshape_add_reshape_transpose_reshape13
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[16], 'i'), (nodes[175], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 479. p167
    nodes[176] = addToKV(176, params['p167'])

    # 480. fused_nn_batch_matmul_368
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[176], 'i'), (nodes[8], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 481. p168
    nodes[177] = addToKV(177, params['p168'])

    # 482. fused_reshape_add_reshape_transpose_reshape_transpose10
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[8], 'i'), (nodes[177], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 483. fused_nn_batch_matmul_513
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[16], 'i'), (nodes[8], 't')]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 484. fused_reshape_divide_add13
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[8], 'i'), (nodes[7], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 485. fused_max10
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 486. fused_subtract_exp13
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[6], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 487. fused_sum10
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 488. fused_divide_reshape13
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[16], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 489. p169
    nodes[178] = addToKV(178, params['p169'])

    # 490. fused_nn_batch_matmul_369
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[178], 'i'), (nodes[6], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 491. p170
    nodes[179] = addToKV(179, params['p170'])

    # 492. fused_reshape_add_reshape_transpose_reshape_transpose_110
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[6], 'i'), (nodes[179], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 493. fused_nn_batch_matmul_413
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[16], 'i'), (nodes[6], 't')]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 494. fused_reshape_transpose_reshape13
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[6], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 495. p171
    nodes[180] = addToKV(180, params['p171'])

    # 496. fused_nn_batch_matmul_326
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[180], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 497. p172
    nodes[181] = addToKV(181, params['p172'])

    # 498. fused_reshape_add_add27
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[16], 'i'), (nodes[181], 'i'), (nodes[13], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 499. fused_mean21
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a42', output_size, const=False, ephemeral=True))
    arguments = [(nodes[6], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 500. fused_subtract27
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[6], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 501. fused_power_mean27
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a43', output_size, const=False, ephemeral=True))
    arguments = [(nodes[8], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 502. p173
    nodes[182] = addToKV(182, params['p173'])

    # 503. p174
    nodes[183] = addToKV(183, params['p174'])

    # 504. fused_add_sqrt_divide_multiply_add26
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[17], 'i'), (nodes[8], 'i'), (nodes[182], 'i'), (nodes[183], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 505. reshape_nop
    nodes[16] = nodes[16]

    # 506. p175
    nodes[184] = addToKV(184, params['p175'])

    # 507. fused_nn_batch_matmul_213
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[184], 'i'), (nodes[13], 't')]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 508. p176
    nodes[185] = addToKV(185, params['p176'])

    # 509. fused_reshape_add_multiply_divide_erf_add_multiply_reshape13
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[13], 'i'), (nodes[185], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 510. p177
    nodes[186] = addToKV(186, params['p177'])

    # 511. fused_nn_batch_matmul_113
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[186], 'i'), (nodes[8], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 512. p178
    nodes[187] = addToKV(187, params['p178'])

    # 513. fused_reshape_add_add26
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[8], 'i'), (nodes[187], 'i'), (nodes[16], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 514. fused_mean22
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a44', output_size, const=False, ephemeral=True))
    arguments = [(nodes[13], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 515. fused_subtract26
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[13], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 516. fused_power_mean26
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a45', output_size, const=False, ephemeral=True))
    arguments = [(nodes[6], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 517. p179
    nodes[188] = addToKV(188, params['p179'])

    # 518. p180
    nodes[189] = addToKV(189, params['p180'])

    # 519. fused_add_sqrt_divide_multiply_add25
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[17], 'i'), (nodes[6], 'i'), (nodes[188], 'i'), (nodes[189], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 520. reshape_nop
    nodes[8] = nodes[8]

    # 521. p181
    nodes[190] = addToKV(190, params['p181'])

    # 522. fused_nn_batch_matmul_325
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[190], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 523. p182
    nodes[191] = addToKV(191, params['p182'])

    # 524. fused_reshape_add_reshape_transpose_reshape12
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[16], 'i'), (nodes[191], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 525. p183
    nodes[192] = addToKV(192, params['p183'])

    # 526. fused_nn_batch_matmul_370
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[192], 'i'), (nodes[6], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 527. p184
    nodes[193] = addToKV(193, params['p184'])

    # 528. fused_reshape_add_reshape_transpose_reshape_transpose11
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[6], 'i'), (nodes[193], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 529. fused_nn_batch_matmul_512
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[16], 'i'), (nodes[6], 't')]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 530. fused_reshape_divide_add12
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[6], 'i'), (nodes[7], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 531. fused_max11
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 532. fused_subtract_exp12
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[13], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 533. fused_sum11
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 534. fused_divide_reshape12
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[16], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 535. p185
    nodes[194] = addToKV(194, params['p185'])

    # 536. fused_nn_batch_matmul_371
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[194], 'i'), (nodes[13], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 537. p186
    nodes[195] = addToKV(195, params['p186'])

    # 538. fused_reshape_add_reshape_transpose_reshape_transpose_111
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[13], 'i'), (nodes[195], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 539. fused_nn_batch_matmul_412
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[16], 'i'), (nodes[13], 't')]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 540. fused_reshape_transpose_reshape12
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[13], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 541. p187
    nodes[196] = addToKV(196, params['p187'])

    # 542. fused_nn_batch_matmul_324
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[196], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 543. p188
    nodes[197] = addToKV(197, params['p188'])

    # 544. fused_reshape_add_add25
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[16], 'i'), (nodes[197], 'i'), (nodes[8], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 545. fused_mean23
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a46', output_size, const=False, ephemeral=True))
    arguments = [(nodes[13], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 546. fused_subtract25
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[13], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 547. fused_power_mean25
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a47', output_size, const=False, ephemeral=True))
    arguments = [(nodes[6], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 548. p189
    nodes[198] = addToKV(198, params['p189'])

    # 549. p190
    nodes[199] = addToKV(199, params['p190'])

    # 550. fused_add_sqrt_divide_multiply_add24
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[17], 'i'), (nodes[6], 'i'), (nodes[198], 'i'), (nodes[199], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 551. reshape_nop
    nodes[16] = nodes[16]

    # 552. p191
    nodes[200] = addToKV(200, params['p191'])

    # 553. fused_nn_batch_matmul_212
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[200], 'i'), (nodes[8], 't')]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 554. p192
    nodes[201] = addToKV(201, params['p192'])

    # 555. fused_reshape_add_multiply_divide_erf_add_multiply_reshape12
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[8], 'i'), (nodes[201], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 556. p193
    nodes[202] = addToKV(202, params['p193'])

    # 557. fused_nn_batch_matmul_112
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[202], 'i'), (nodes[6], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 558. p194
    nodes[203] = addToKV(203, params['p194'])

    # 559. fused_reshape_add_add24
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[6], 'i'), (nodes[203], 'i'), (nodes[16], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 560. fused_mean24
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a48', output_size, const=False, ephemeral=True))
    arguments = [(nodes[8], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 561. fused_subtract24
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[8], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 562. fused_power_mean24
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a49', output_size, const=False, ephemeral=True))
    arguments = [(nodes[13], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 563. p195
    nodes[204] = addToKV(204, params['p195'])

    # 564. p196
    nodes[205] = addToKV(205, params['p196'])

    # 565. fused_add_sqrt_divide_multiply_add23
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[17], 'i'), (nodes[13], 'i'), (nodes[204], 'i'), (nodes[205], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 566. reshape_nop
    nodes[6] = nodes[6]

    # 567. p197
    nodes[206] = addToKV(206, params['p197'])

    # 568. fused_nn_batch_matmul_323
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[206], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 569. p198
    nodes[207] = addToKV(207, params['p198'])

    # 570. fused_reshape_add_reshape_transpose_reshape11
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[16], 'i'), (nodes[207], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 571. p199
    nodes[208] = addToKV(208, params['p199'])

    # 572. fused_nn_batch_matmul_372
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[208], 'i'), (nodes[13], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 573. p200
    nodes[209] = addToKV(209, params['p200'])

    # 574. fused_reshape_add_reshape_transpose_reshape_transpose12
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[13], 'i'), (nodes[209], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 575. fused_nn_batch_matmul_511
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[16], 'i'), (nodes[13], 't')]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 576. fused_reshape_divide_add11
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[13], 'i'), (nodes[7], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 577. fused_max12
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 578. fused_subtract_exp11
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[8], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 579. fused_sum12
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 580. fused_divide_reshape11
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[16], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 581. p201
    nodes[210] = addToKV(210, params['p201'])

    # 582. fused_nn_batch_matmul_373
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[210], 'i'), (nodes[8], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 583. p202
    nodes[211] = addToKV(211, params['p202'])

    # 584. fused_reshape_add_reshape_transpose_reshape_transpose_112
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[8], 'i'), (nodes[211], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 585. fused_nn_batch_matmul_411
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[16], 'i'), (nodes[8], 't')]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 586. fused_reshape_transpose_reshape11
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[8], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 587. p203
    nodes[212] = addToKV(212, params['p203'])

    # 588. fused_nn_batch_matmul_322
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[212], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 589. p204
    nodes[213] = addToKV(213, params['p204'])

    # 590. fused_reshape_add_add23
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[16], 'i'), (nodes[213], 'i'), (nodes[6], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 591. fused_mean25
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a50', output_size, const=False, ephemeral=True))
    arguments = [(nodes[8], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 592. fused_subtract23
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[8], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 593. fused_power_mean23
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a51', output_size, const=False, ephemeral=True))
    arguments = [(nodes[13], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 594. p205
    nodes[214] = addToKV(214, params['p205'])

    # 595. p206
    nodes[215] = addToKV(215, params['p206'])

    # 596. fused_add_sqrt_divide_multiply_add22
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[17], 'i'), (nodes[13], 'i'), (nodes[214], 'i'), (nodes[215], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 597. reshape_nop
    nodes[16] = nodes[16]

    # 598. p207
    nodes[216] = addToKV(216, params['p207'])

    # 599. fused_nn_batch_matmul_211
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[216], 'i'), (nodes[6], 't')]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 600. p208
    nodes[217] = addToKV(217, params['p208'])

    # 601. fused_reshape_add_multiply_divide_erf_add_multiply_reshape11
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[6], 'i'), (nodes[217], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 602. p209
    nodes[218] = addToKV(218, params['p209'])

    # 603. fused_nn_batch_matmul_111
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[218], 'i'), (nodes[13], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 604. p210
    nodes[219] = addToKV(219, params['p210'])

    # 605. fused_reshape_add_add22
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[13], 'i'), (nodes[219], 'i'), (nodes[16], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 606. fused_mean26
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a52', output_size, const=False, ephemeral=True))
    arguments = [(nodes[6], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 607. fused_subtract22
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[6], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 608. fused_power_mean22
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a53', output_size, const=False, ephemeral=True))
    arguments = [(nodes[8], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 609. p211
    nodes[220] = addToKV(220, params['p211'])

    # 610. p212
    nodes[221] = addToKV(221, params['p212'])

    # 611. fused_add_sqrt_divide_multiply_add21
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[17], 'i'), (nodes[8], 'i'), (nodes[220], 'i'), (nodes[221], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 612. reshape_nop
    nodes[13] = nodes[13]

    # 613. p213
    nodes[222] = addToKV(222, params['p213'])

    # 614. fused_nn_batch_matmul_321
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[222], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 615. p214
    nodes[223] = addToKV(223, params['p214'])

    # 616. fused_reshape_add_reshape_transpose_reshape10
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[16], 'i'), (nodes[223], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 617. p215
    nodes[224] = addToKV(224, params['p215'])

    # 618. fused_nn_batch_matmul_374
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[224], 'i'), (nodes[8], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 619. p216
    nodes[225] = addToKV(225, params['p216'])

    # 620. fused_reshape_add_reshape_transpose_reshape_transpose13
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[8], 'i'), (nodes[225], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 621. fused_nn_batch_matmul_510
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[16], 'i'), (nodes[8], 't')]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 622. fused_reshape_divide_add10
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[8], 'i'), (nodes[7], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 623. fused_max13
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 624. fused_subtract_exp10
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[6], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 625. fused_sum13
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 626. fused_divide_reshape10
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[16], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 627. p217
    nodes[226] = addToKV(226, params['p217'])

    # 628. fused_nn_batch_matmul_375
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[226], 'i'), (nodes[6], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 629. p218
    nodes[227] = addToKV(227, params['p218'])

    # 630. fused_reshape_add_reshape_transpose_reshape_transpose_113
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[6], 'i'), (nodes[227], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 631. fused_nn_batch_matmul_410
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[16], 'i'), (nodes[6], 't')]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 632. fused_reshape_transpose_reshape10
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[6], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 633. p219
    nodes[228] = addToKV(228, params['p219'])

    # 634. fused_nn_batch_matmul_320
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[228], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 635. p220
    nodes[229] = addToKV(229, params['p220'])

    # 636. fused_reshape_add_add21
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[16], 'i'), (nodes[229], 'i'), (nodes[13], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 637. fused_mean27
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a54', output_size, const=False, ephemeral=True))
    arguments = [(nodes[6], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 638. fused_subtract21
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[6], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 639. fused_power_mean21
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a55', output_size, const=False, ephemeral=True))
    arguments = [(nodes[8], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 640. p221
    nodes[230] = addToKV(230, params['p221'])

    # 641. p222
    nodes[231] = addToKV(231, params['p222'])

    # 642. fused_add_sqrt_divide_multiply_add20
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[17], 'i'), (nodes[8], 'i'), (nodes[230], 'i'), (nodes[231], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 643. reshape_nop
    nodes[16] = nodes[16]

    # 644. p223
    nodes[232] = addToKV(232, params['p223'])

    # 645. fused_nn_batch_matmul_210
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[232], 'i'), (nodes[13], 't')]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 646. p224
    nodes[233] = addToKV(233, params['p224'])

    # 647. fused_reshape_add_multiply_divide_erf_add_multiply_reshape10
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[13], 'i'), (nodes[233], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 648. p225
    nodes[234] = addToKV(234, params['p225'])

    # 649. fused_nn_batch_matmul_110
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[234], 'i'), (nodes[8], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 650. p226
    nodes[235] = addToKV(235, params['p226'])

    # 651. fused_reshape_add_add20
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[8], 'i'), (nodes[235], 'i'), (nodes[16], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 652. fused_mean28
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a56', output_size, const=False, ephemeral=True))
    arguments = [(nodes[13], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 653. fused_subtract20
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[13], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 654. fused_power_mean20
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a57', output_size, const=False, ephemeral=True))
    arguments = [(nodes[6], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 655. p227
    nodes[236] = addToKV(236, params['p227'])

    # 656. p228
    nodes[237] = addToKV(237, params['p228'])

    # 657. fused_add_sqrt_divide_multiply_add19
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[17], 'i'), (nodes[6], 'i'), (nodes[236], 'i'), (nodes[237], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 658. reshape_nop
    nodes[8] = nodes[8]

    # 659. p229
    nodes[238] = addToKV(238, params['p229'])

    # 660. fused_nn_batch_matmul_319
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[238], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 661. p230
    nodes[239] = addToKV(239, params['p230'])

    # 662. fused_reshape_add_reshape_transpose_reshape9
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[16], 'i'), (nodes[239], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 663. p231
    nodes[240] = addToKV(240, params['p231'])

    # 664. fused_nn_batch_matmul_376
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[240], 'i'), (nodes[6], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 665. p232
    nodes[241] = addToKV(241, params['p232'])

    # 666. fused_reshape_add_reshape_transpose_reshape_transpose14
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[6], 'i'), (nodes[241], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 667. fused_nn_batch_matmul_59
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[16], 'i'), (nodes[6], 't')]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 668. fused_reshape_divide_add9
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[6], 'i'), (nodes[7], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 669. fused_max14
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 670. fused_subtract_exp9
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[13], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 671. fused_sum14
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 672. fused_divide_reshape9
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[16], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 673. p233
    nodes[242] = addToKV(242, params['p233'])

    # 674. fused_nn_batch_matmul_377
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[242], 'i'), (nodes[13], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 675. p234
    nodes[243] = addToKV(243, params['p234'])

    # 676. fused_reshape_add_reshape_transpose_reshape_transpose_114
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[13], 'i'), (nodes[243], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 677. fused_nn_batch_matmul_49
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[16], 'i'), (nodes[13], 't')]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 678. fused_reshape_transpose_reshape9
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[13], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 679. p235
    nodes[244] = addToKV(244, params['p235'])

    # 680. fused_nn_batch_matmul_318
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[244], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 681. p236
    nodes[245] = addToKV(245, params['p236'])

    # 682. fused_reshape_add_add19
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[16], 'i'), (nodes[245], 'i'), (nodes[8], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 683. fused_mean29
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a58', output_size, const=False, ephemeral=True))
    arguments = [(nodes[13], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 684. fused_subtract19
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[13], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 685. fused_power_mean19
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a59', output_size, const=False, ephemeral=True))
    arguments = [(nodes[6], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 686. p237
    nodes[246] = addToKV(246, params['p237'])

    # 687. p238
    nodes[247] = addToKV(247, params['p238'])

    # 688. fused_add_sqrt_divide_multiply_add18
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[17], 'i'), (nodes[6], 'i'), (nodes[246], 'i'), (nodes[247], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 689. reshape_nop
    nodes[16] = nodes[16]

    # 690. p239
    nodes[248] = addToKV(248, params['p239'])

    # 691. fused_nn_batch_matmul_29
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[248], 'i'), (nodes[8], 't')]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 692. p240
    nodes[249] = addToKV(249, params['p240'])

    # 693. fused_reshape_add_multiply_divide_erf_add_multiply_reshape9
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[8], 'i'), (nodes[249], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 694. p241
    nodes[250] = addToKV(250, params['p241'])

    # 695. fused_nn_batch_matmul_19
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[250], 'i'), (nodes[6], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 696. p242
    nodes[251] = addToKV(251, params['p242'])

    # 697. fused_reshape_add_add18
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[6], 'i'), (nodes[251], 'i'), (nodes[16], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 698. fused_mean30
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a60', output_size, const=False, ephemeral=True))
    arguments = [(nodes[8], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 699. fused_subtract18
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[8], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 700. fused_power_mean18
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a61', output_size, const=False, ephemeral=True))
    arguments = [(nodes[13], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 701. p243
    nodes[252] = addToKV(252, params['p243'])

    # 702. p244
    nodes[253] = addToKV(253, params['p244'])

    # 703. fused_add_sqrt_divide_multiply_add17
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[17], 'i'), (nodes[13], 'i'), (nodes[252], 'i'), (nodes[253], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 704. reshape_nop
    nodes[6] = nodes[6]

    # 705. p245
    nodes[254] = addToKV(254, params['p245'])

    # 706. fused_nn_batch_matmul_317
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[254], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 707. p246
    nodes[255] = addToKV(255, params['p246'])

    # 708. fused_reshape_add_reshape_transpose_reshape8
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[16], 'i'), (nodes[255], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 709. p247
    nodes[256] = addToKV(256, params['p247'])

    # 710. fused_nn_batch_matmul_378
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[256], 'i'), (nodes[13], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 711. p248
    nodes[257] = addToKV(257, params['p248'])

    # 712. fused_reshape_add_reshape_transpose_reshape_transpose15
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[13], 'i'), (nodes[257], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 713. fused_nn_batch_matmul_58
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[16], 'i'), (nodes[13], 't')]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 714. fused_reshape_divide_add8
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[13], 'i'), (nodes[7], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 715. fused_max15
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 716. fused_subtract_exp8
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[8], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 717. fused_sum15
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 718. fused_divide_reshape8
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[16], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 719. p249
    nodes[258] = addToKV(258, params['p249'])

    # 720. fused_nn_batch_matmul_379
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[258], 'i'), (nodes[8], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 721. p250
    nodes[259] = addToKV(259, params['p250'])

    # 722. fused_reshape_add_reshape_transpose_reshape_transpose_115
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[8], 'i'), (nodes[259], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 723. fused_nn_batch_matmul_48
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[16], 'i'), (nodes[8], 't')]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 724. fused_reshape_transpose_reshape8
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[8], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 725. p251
    nodes[260] = addToKV(260, params['p251'])

    # 726. fused_nn_batch_matmul_316
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[260], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 727. p252
    nodes[261] = addToKV(261, params['p252'])

    # 728. fused_reshape_add_add17
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[16], 'i'), (nodes[261], 'i'), (nodes[6], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 729. fused_mean31
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a62', output_size, const=False, ephemeral=True))
    arguments = [(nodes[8], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 730. fused_subtract17
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[8], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 731. fused_power_mean17
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a63', output_size, const=False, ephemeral=True))
    arguments = [(nodes[13], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 732. p253
    nodes[262] = addToKV(262, params['p253'])

    # 733. p254
    nodes[263] = addToKV(263, params['p254'])

    # 734. fused_add_sqrt_divide_multiply_add16
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[17], 'i'), (nodes[13], 'i'), (nodes[262], 'i'), (nodes[263], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 735. reshape_nop
    nodes[16] = nodes[16]

    # 736. p255
    nodes[264] = addToKV(264, params['p255'])

    # 737. fused_nn_batch_matmul_28
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[264], 'i'), (nodes[6], 't')]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 738. p256
    nodes[265] = addToKV(265, params['p256'])

    # 739. fused_reshape_add_multiply_divide_erf_add_multiply_reshape8
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[6], 'i'), (nodes[265], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 740. p257
    nodes[266] = addToKV(266, params['p257'])

    # 741. fused_nn_batch_matmul_18
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[266], 'i'), (nodes[13], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 742. p258
    nodes[267] = addToKV(267, params['p258'])

    # 743. fused_reshape_add_add16
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[13], 'i'), (nodes[267], 'i'), (nodes[16], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 744. fused_mean32
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a64', output_size, const=False, ephemeral=True))
    arguments = [(nodes[6], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 745. fused_subtract16
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[6], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 746. fused_power_mean16
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a65', output_size, const=False, ephemeral=True))
    arguments = [(nodes[8], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 747. p259
    nodes[268] = addToKV(268, params['p259'])

    # 748. p260
    nodes[269] = addToKV(269, params['p260'])

    # 749. fused_add_sqrt_divide_multiply_add15
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[17], 'i'), (nodes[8], 'i'), (nodes[268], 'i'), (nodes[269], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 750. reshape_nop
    nodes[13] = nodes[13]

    # 751. p261
    nodes[270] = addToKV(270, params['p261'])

    # 752. fused_nn_batch_matmul_315
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[270], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 753. p262
    nodes[271] = addToKV(271, params['p262'])

    # 754. fused_reshape_add_reshape_transpose_reshape7
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[16], 'i'), (nodes[271], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 755. p263
    nodes[272] = addToKV(272, params['p263'])

    # 756. fused_nn_batch_matmul_380
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[272], 'i'), (nodes[8], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 757. p264
    nodes[273] = addToKV(273, params['p264'])

    # 758. fused_reshape_add_reshape_transpose_reshape_transpose16
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[8], 'i'), (nodes[273], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 759. fused_nn_batch_matmul_57
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[16], 'i'), (nodes[8], 't')]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 760. fused_reshape_divide_add7
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[8], 'i'), (nodes[7], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 761. fused_max16
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 762. fused_subtract_exp7
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[6], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 763. fused_sum16
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 764. fused_divide_reshape7
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[16], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 765. p265
    nodes[274] = addToKV(274, params['p265'])

    # 766. fused_nn_batch_matmul_381
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[274], 'i'), (nodes[6], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 767. p266
    nodes[275] = addToKV(275, params['p266'])

    # 768. fused_reshape_add_reshape_transpose_reshape_transpose_116
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[6], 'i'), (nodes[275], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 769. fused_nn_batch_matmul_47
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[16], 'i'), (nodes[6], 't')]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 770. fused_reshape_transpose_reshape7
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[6], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 771. p267
    nodes[276] = addToKV(276, params['p267'])

    # 772. fused_nn_batch_matmul_314
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[276], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 773. p268
    nodes[277] = addToKV(277, params['p268'])

    # 774. fused_reshape_add_add15
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[16], 'i'), (nodes[277], 'i'), (nodes[13], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 775. fused_mean33
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a66', output_size, const=False, ephemeral=True))
    arguments = [(nodes[6], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 776. fused_subtract15
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[6], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 777. fused_power_mean15
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a67', output_size, const=False, ephemeral=True))
    arguments = [(nodes[8], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 778. p269
    nodes[278] = addToKV(278, params['p269'])

    # 779. p270
    nodes[279] = addToKV(279, params['p270'])

    # 780. fused_add_sqrt_divide_multiply_add14
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[17], 'i'), (nodes[8], 'i'), (nodes[278], 'i'), (nodes[279], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 781. reshape_nop
    nodes[16] = nodes[16]

    # 782. p271
    nodes[280] = addToKV(280, params['p271'])

    # 783. fused_nn_batch_matmul_27
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[280], 'i'), (nodes[13], 't')]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 784. p272
    nodes[281] = addToKV(281, params['p272'])

    # 785. fused_reshape_add_multiply_divide_erf_add_multiply_reshape7
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[13], 'i'), (nodes[281], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 786. p273
    nodes[282] = addToKV(282, params['p273'])

    # 787. fused_nn_batch_matmul_17
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[282], 'i'), (nodes[8], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 788. p274
    nodes[283] = addToKV(283, params['p274'])

    # 789. fused_reshape_add_add14
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[8], 'i'), (nodes[283], 'i'), (nodes[16], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 790. fused_mean34
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a68', output_size, const=False, ephemeral=True))
    arguments = [(nodes[13], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 791. fused_subtract14
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[13], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 792. fused_power_mean14
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a69', output_size, const=False, ephemeral=True))
    arguments = [(nodes[6], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 793. p275
    nodes[284] = addToKV(284, params['p275'])

    # 794. p276
    nodes[285] = addToKV(285, params['p276'])

    # 795. fused_add_sqrt_divide_multiply_add13
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[17], 'i'), (nodes[6], 'i'), (nodes[284], 'i'), (nodes[285], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 796. reshape_nop
    nodes[8] = nodes[8]

    # 797. p277
    nodes[286] = addToKV(286, params['p277'])

    # 798. fused_nn_batch_matmul_313
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[286], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 799. p278
    nodes[287] = addToKV(287, params['p278'])

    # 800. fused_reshape_add_reshape_transpose_reshape6
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[16], 'i'), (nodes[287], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 801. p279
    nodes[288] = addToKV(288, params['p279'])

    # 802. fused_nn_batch_matmul_382
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[288], 'i'), (nodes[6], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 803. p280
    nodes[289] = addToKV(289, params['p280'])

    # 804. fused_reshape_add_reshape_transpose_reshape_transpose17
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[6], 'i'), (nodes[289], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 805. fused_nn_batch_matmul_56
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[16], 'i'), (nodes[6], 't')]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 806. fused_reshape_divide_add6
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[6], 'i'), (nodes[7], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 807. fused_max17
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 808. fused_subtract_exp6
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[13], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 809. fused_sum17
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 810. fused_divide_reshape6
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[16], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 811. p281
    nodes[290] = addToKV(290, params['p281'])

    # 812. fused_nn_batch_matmul_383
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[290], 'i'), (nodes[13], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 813. p282
    nodes[291] = addToKV(291, params['p282'])

    # 814. fused_reshape_add_reshape_transpose_reshape_transpose_117
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[13], 'i'), (nodes[291], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 815. fused_nn_batch_matmul_46
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[16], 'i'), (nodes[13], 't')]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 816. fused_reshape_transpose_reshape6
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[13], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 817. p283
    nodes[292] = addToKV(292, params['p283'])

    # 818. fused_nn_batch_matmul_312
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[292], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 819. p284
    nodes[293] = addToKV(293, params['p284'])

    # 820. fused_reshape_add_add13
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[16], 'i'), (nodes[293], 'i'), (nodes[8], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 821. fused_mean35
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a70', output_size, const=False, ephemeral=True))
    arguments = [(nodes[13], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 822. fused_subtract13
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[13], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 823. fused_power_mean13
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a71', output_size, const=False, ephemeral=True))
    arguments = [(nodes[6], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 824. p285
    nodes[294] = addToKV(294, params['p285'])

    # 825. p286
    nodes[295] = addToKV(295, params['p286'])

    # 826. fused_add_sqrt_divide_multiply_add12
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[17], 'i'), (nodes[6], 'i'), (nodes[294], 'i'), (nodes[295], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 827. reshape_nop
    nodes[16] = nodes[16]

    # 828. p287
    nodes[296] = addToKV(296, params['p287'])

    # 829. fused_nn_batch_matmul_26
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[296], 'i'), (nodes[8], 't')]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 830. p288
    nodes[297] = addToKV(297, params['p288'])

    # 831. fused_reshape_add_multiply_divide_erf_add_multiply_reshape6
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[8], 'i'), (nodes[297], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 832. p289
    nodes[298] = addToKV(298, params['p289'])

    # 833. fused_nn_batch_matmul_16
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[298], 'i'), (nodes[6], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 834. p290
    nodes[299] = addToKV(299, params['p290'])

    # 835. fused_reshape_add_add12
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[6], 'i'), (nodes[299], 'i'), (nodes[16], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 836. fused_mean36
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a72', output_size, const=False, ephemeral=True))
    arguments = [(nodes[8], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 837. fused_subtract12
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[8], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 838. fused_power_mean12
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a73', output_size, const=False, ephemeral=True))
    arguments = [(nodes[13], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 839. p291
    nodes[300] = addToKV(300, params['p291'])

    # 840. p292
    nodes[301] = addToKV(301, params['p292'])

    # 841. fused_add_sqrt_divide_multiply_add11
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[17], 'i'), (nodes[13], 'i'), (nodes[300], 'i'), (nodes[301], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 842. reshape_nop
    nodes[6] = nodes[6]

    # 843. p293
    nodes[302] = addToKV(302, params['p293'])

    # 844. fused_nn_batch_matmul_311
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[302], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 845. p294
    nodes[303] = addToKV(303, params['p294'])

    # 846. fused_reshape_add_reshape_transpose_reshape5
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[16], 'i'), (nodes[303], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 847. p295
    nodes[304] = addToKV(304, params['p295'])

    # 848. fused_nn_batch_matmul_384
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[304], 'i'), (nodes[13], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 849. p296
    nodes[305] = addToKV(305, params['p296'])

    # 850. fused_reshape_add_reshape_transpose_reshape_transpose18
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[13], 'i'), (nodes[305], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 851. fused_nn_batch_matmul_55
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[16], 'i'), (nodes[13], 't')]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 852. fused_reshape_divide_add5
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[13], 'i'), (nodes[7], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 853. fused_max18
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 854. fused_subtract_exp5
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[8], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 855. fused_sum18
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 856. fused_divide_reshape5
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[16], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 857. p297
    nodes[306] = addToKV(306, params['p297'])

    # 858. fused_nn_batch_matmul_385
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[306], 'i'), (nodes[8], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 859. p298
    nodes[307] = addToKV(307, params['p298'])

    # 860. fused_reshape_add_reshape_transpose_reshape_transpose_118
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[8], 'i'), (nodes[307], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 861. fused_nn_batch_matmul_45
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[16], 'i'), (nodes[8], 't')]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 862. fused_reshape_transpose_reshape5
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[8], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 863. p299
    nodes[308] = addToKV(308, params['p299'])

    # 864. fused_nn_batch_matmul_310
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[308], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 865. p300
    nodes[309] = addToKV(309, params['p300'])

    # 866. fused_reshape_add_add11
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[16], 'i'), (nodes[309], 'i'), (nodes[6], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 867. fused_mean37
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a74', output_size, const=False, ephemeral=True))
    arguments = [(nodes[8], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 868. fused_subtract11
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[8], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 869. fused_power_mean11
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a75', output_size, const=False, ephemeral=True))
    arguments = [(nodes[13], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 870. p301
    nodes[310] = addToKV(310, params['p301'])

    # 871. p302
    nodes[311] = addToKV(311, params['p302'])

    # 872. fused_add_sqrt_divide_multiply_add10
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[17], 'i'), (nodes[13], 'i'), (nodes[310], 'i'), (nodes[311], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 873. reshape_nop
    nodes[16] = nodes[16]

    # 874. p303
    nodes[312] = addToKV(312, params['p303'])

    # 875. fused_nn_batch_matmul_25
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[312], 'i'), (nodes[6], 't')]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 876. p304
    nodes[313] = addToKV(313, params['p304'])

    # 877. fused_reshape_add_multiply_divide_erf_add_multiply_reshape5
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[6], 'i'), (nodes[313], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 878. p305
    nodes[314] = addToKV(314, params['p305'])

    # 879. fused_nn_batch_matmul_15
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[314], 'i'), (nodes[13], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 880. p306
    nodes[315] = addToKV(315, params['p306'])

    # 881. fused_reshape_add_add10
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[13], 'i'), (nodes[315], 'i'), (nodes[16], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 882. fused_mean38
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a76', output_size, const=False, ephemeral=True))
    arguments = [(nodes[6], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 883. fused_subtract10
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[6], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 884. fused_power_mean10
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a77', output_size, const=False, ephemeral=True))
    arguments = [(nodes[8], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 885. p307
    nodes[316] = addToKV(316, params['p307'])

    # 886. p308
    nodes[317] = addToKV(317, params['p308'])

    # 887. fused_add_sqrt_divide_multiply_add9
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[17], 'i'), (nodes[8], 'i'), (nodes[316], 'i'), (nodes[317], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 888. reshape_nop
    nodes[13] = nodes[13]

    # 889. p309
    nodes[318] = addToKV(318, params['p309'])

    # 890. fused_nn_batch_matmul_39
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[318], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 891. p310
    nodes[319] = addToKV(319, params['p310'])

    # 892. fused_reshape_add_reshape_transpose_reshape4
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[16], 'i'), (nodes[319], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 893. p311
    nodes[320] = addToKV(320, params['p311'])

    # 894. fused_nn_batch_matmul_386
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[320], 'i'), (nodes[8], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 895. p312
    nodes[321] = addToKV(321, params['p312'])

    # 896. fused_reshape_add_reshape_transpose_reshape_transpose19
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[8], 'i'), (nodes[321], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 897. fused_nn_batch_matmul_54
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[16], 'i'), (nodes[8], 't')]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 898. fused_reshape_divide_add4
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[8], 'i'), (nodes[7], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 899. fused_max19
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 900. fused_subtract_exp4
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[6], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 901. fused_sum19
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 902. fused_divide_reshape4
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[16], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 903. p313
    nodes[322] = addToKV(322, params['p313'])

    # 904. fused_nn_batch_matmul_387
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[322], 'i'), (nodes[6], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 905. p314
    nodes[323] = addToKV(323, params['p314'])

    # 906. fused_reshape_add_reshape_transpose_reshape_transpose_119
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[6], 'i'), (nodes[323], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 907. fused_nn_batch_matmul_44
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[16], 'i'), (nodes[6], 't')]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 908. fused_reshape_transpose_reshape4
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[6], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 909. p315
    nodes[324] = addToKV(324, params['p315'])

    # 910. fused_nn_batch_matmul_38
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[324], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 911. p316
    nodes[325] = addToKV(325, params['p316'])

    # 912. fused_reshape_add_add9
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[16], 'i'), (nodes[325], 'i'), (nodes[13], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 913. fused_mean39
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a78', output_size, const=False, ephemeral=True))
    arguments = [(nodes[6], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 914. fused_subtract9
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[6], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 915. fused_power_mean9
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a79', output_size, const=False, ephemeral=True))
    arguments = [(nodes[8], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 916. p317
    nodes[326] = addToKV(326, params['p317'])

    # 917. p318
    nodes[327] = addToKV(327, params['p318'])

    # 918. fused_add_sqrt_divide_multiply_add8
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[17], 'i'), (nodes[8], 'i'), (nodes[326], 'i'), (nodes[327], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 919. reshape_nop
    nodes[16] = nodes[16]

    # 920. p319
    nodes[328] = addToKV(328, params['p319'])

    # 921. fused_nn_batch_matmul_24
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[328], 'i'), (nodes[13], 't')]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 922. p320
    nodes[329] = addToKV(329, params['p320'])

    # 923. fused_reshape_add_multiply_divide_erf_add_multiply_reshape4
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[13], 'i'), (nodes[329], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 924. p321
    nodes[330] = addToKV(330, params['p321'])

    # 925. fused_nn_batch_matmul_14
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[330], 'i'), (nodes[8], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 926. p322
    nodes[331] = addToKV(331, params['p322'])

    # 927. fused_reshape_add_add8
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[8], 'i'), (nodes[331], 'i'), (nodes[16], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 928. fused_mean40
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a80', output_size, const=False, ephemeral=True))
    arguments = [(nodes[13], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 929. fused_subtract8
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[13], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 930. fused_power_mean8
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a81', output_size, const=False, ephemeral=True))
    arguments = [(nodes[6], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 931. p323
    nodes[332] = addToKV(332, params['p323'])

    # 932. p324
    nodes[333] = addToKV(333, params['p324'])

    # 933. fused_add_sqrt_divide_multiply_add7
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[17], 'i'), (nodes[6], 'i'), (nodes[332], 'i'), (nodes[333], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 934. reshape_nop
    nodes[8] = nodes[8]

    # 935. p325
    nodes[334] = addToKV(334, params['p325'])

    # 936. fused_nn_batch_matmul_37
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[334], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 937. p326
    nodes[335] = addToKV(335, params['p326'])

    # 938. fused_reshape_add_reshape_transpose_reshape3
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[16], 'i'), (nodes[335], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 939. p327
    nodes[336] = addToKV(336, params['p327'])

    # 940. fused_nn_batch_matmul_388
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[336], 'i'), (nodes[6], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 941. p328
    nodes[337] = addToKV(337, params['p328'])

    # 942. fused_reshape_add_reshape_transpose_reshape_transpose20
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[6], 'i'), (nodes[337], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 943. fused_nn_batch_matmul_53
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[16], 'i'), (nodes[6], 't')]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 944. fused_reshape_divide_add3
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[6], 'i'), (nodes[7], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 945. fused_max20
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 946. fused_subtract_exp3
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[13], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 947. fused_sum20
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 948. fused_divide_reshape3
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[16], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 949. p329
    nodes[338] = addToKV(338, params['p329'])

    # 950. fused_nn_batch_matmul_389
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[338], 'i'), (nodes[13], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 951. p330
    nodes[339] = addToKV(339, params['p330'])

    # 952. fused_reshape_add_reshape_transpose_reshape_transpose_120
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[13], 'i'), (nodes[339], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 953. fused_nn_batch_matmul_43
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[16], 'i'), (nodes[13], 't')]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 954. fused_reshape_transpose_reshape3
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[13], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 955. p331
    nodes[340] = addToKV(340, params['p331'])

    # 956. fused_nn_batch_matmul_36
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[340], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 957. p332
    nodes[341] = addToKV(341, params['p332'])

    # 958. fused_reshape_add_add7
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[16], 'i'), (nodes[341], 'i'), (nodes[8], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 959. fused_mean41
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a82', output_size, const=False, ephemeral=True))
    arguments = [(nodes[13], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 960. fused_subtract7
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[13], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 961. fused_power_mean7
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a83', output_size, const=False, ephemeral=True))
    arguments = [(nodes[6], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 962. p333
    nodes[342] = addToKV(342, params['p333'])

    # 963. p334
    nodes[343] = addToKV(343, params['p334'])

    # 964. fused_add_sqrt_divide_multiply_add6
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[17], 'i'), (nodes[6], 'i'), (nodes[342], 'i'), (nodes[343], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 965. reshape_nop
    nodes[16] = nodes[16]

    # 966. p335
    nodes[344] = addToKV(344, params['p335'])

    # 967. fused_nn_batch_matmul_23
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[344], 'i'), (nodes[8], 't')]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 968. p336
    nodes[345] = addToKV(345, params['p336'])

    # 969. fused_reshape_add_multiply_divide_erf_add_multiply_reshape3
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[8], 'i'), (nodes[345], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 970. p337
    nodes[346] = addToKV(346, params['p337'])

    # 971. fused_nn_batch_matmul_13
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[346], 'i'), (nodes[6], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 972. p338
    nodes[347] = addToKV(347, params['p338'])

    # 973. fused_reshape_add_add6
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[6], 'i'), (nodes[347], 'i'), (nodes[16], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 974. fused_mean42
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a84', output_size, const=False, ephemeral=True))
    arguments = [(nodes[8], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 975. fused_subtract6
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[8], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 976. fused_power_mean6
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a85', output_size, const=False, ephemeral=True))
    arguments = [(nodes[13], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 977. p339
    nodes[348] = addToKV(348, params['p339'])

    # 978. p340
    nodes[349] = addToKV(349, params['p340'])

    # 979. fused_add_sqrt_divide_multiply_add5
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[17], 'i'), (nodes[13], 'i'), (nodes[348], 'i'), (nodes[349], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 980. reshape_nop
    nodes[6] = nodes[6]

    # 981. p341
    nodes[350] = addToKV(350, params['p341'])

    # 982. fused_nn_batch_matmul_35
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[350], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 983. p342
    nodes[351] = addToKV(351, params['p342'])

    # 984. fused_reshape_add_reshape_transpose_reshape2
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[16], 'i'), (nodes[351], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 985. p343
    nodes[352] = addToKV(352, params['p343'])

    # 986. fused_nn_batch_matmul_390
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[352], 'i'), (nodes[13], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 987. p344
    nodes[353] = addToKV(353, params['p344'])

    # 988. fused_reshape_add_reshape_transpose_reshape_transpose21
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[13], 'i'), (nodes[353], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 989. fused_nn_batch_matmul_52
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[16], 'i'), (nodes[13], 't')]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 990. fused_reshape_divide_add2
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[13], 'i'), (nodes[7], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 991. fused_max21
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 992. fused_subtract_exp2
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[8], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 993. fused_sum21
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 994. fused_divide_reshape2
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[16], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 995. p345
    nodes[354] = addToKV(354, params['p345'])

    # 996. fused_nn_batch_matmul_391
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[354], 'i'), (nodes[8], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 997. p346
    nodes[355] = addToKV(355, params['p346'])

    # 998. fused_reshape_add_reshape_transpose_reshape_transpose_121
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[8], 'i'), (nodes[355], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 999. fused_nn_batch_matmul_42
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[16], 'i'), (nodes[8], 't')]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 1000. fused_reshape_transpose_reshape2
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[8], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 1001. p347
    nodes[356] = addToKV(356, params['p347'])

    # 1002. fused_nn_batch_matmul_34
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[356], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 1003. p348
    nodes[357] = addToKV(357, params['p348'])

    # 1004. fused_reshape_add_add5
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[16], 'i'), (nodes[357], 'i'), (nodes[6], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 1005. fused_mean43
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a86', output_size, const=False, ephemeral=True))
    arguments = [(nodes[8], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 1006. fused_subtract5
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[8], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 1007. fused_power_mean5
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a87', output_size, const=False, ephemeral=True))
    arguments = [(nodes[13], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 1008. p349
    nodes[358] = addToKV(358, params['p349'])

    # 1009. p350
    nodes[359] = addToKV(359, params['p350'])

    # 1010. fused_add_sqrt_divide_multiply_add4
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[17], 'i'), (nodes[13], 'i'), (nodes[358], 'i'), (nodes[359], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 1011. reshape_nop
    nodes[16] = nodes[16]

    # 1012. p351
    nodes[360] = addToKV(360, params['p351'])

    # 1013. fused_nn_batch_matmul_22
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[360], 'i'), (nodes[6], 't')]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 1014. p352
    nodes[361] = addToKV(361, params['p352'])

    # 1015. fused_reshape_add_multiply_divide_erf_add_multiply_reshape2
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[6], 'i'), (nodes[361], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 1016. p353
    nodes[362] = addToKV(362, params['p353'])

    # 1017. fused_nn_batch_matmul_12
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[362], 'i'), (nodes[13], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 1018. p354
    nodes[363] = addToKV(363, params['p354'])

    # 1019. fused_reshape_add_add4
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[13], 'i'), (nodes[363], 'i'), (nodes[16], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 1020. fused_mean44
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a88', output_size, const=False, ephemeral=True))
    arguments = [(nodes[6], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 1021. fused_subtract4
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[6], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 1022. fused_power_mean4
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a89', output_size, const=False, ephemeral=True))
    arguments = [(nodes[8], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 1023. p355
    nodes[364] = addToKV(364, params['p355'])

    # 1024. p356
    nodes[365] = addToKV(365, params['p356'])

    # 1025. fused_add_sqrt_divide_multiply_add3
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[17], 'i'), (nodes[8], 'i'), (nodes[364], 'i'), (nodes[365], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 1026. reshape_nop
    nodes[13] = nodes[13]

    # 1027. p357
    nodes[366] = addToKV(366, params['p357'])

    # 1028. fused_nn_batch_matmul_33
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[366], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 1029. p358
    nodes[367] = addToKV(367, params['p358'])

    # 1030. fused_reshape_add_reshape_transpose_reshape1
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[16], 'i'), (nodes[367], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 1031. p359
    nodes[368] = addToKV(368, params['p359'])

    # 1032. fused_nn_batch_matmul_392
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[368], 'i'), (nodes[8], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 1033. p360
    nodes[369] = addToKV(369, params['p360'])

    # 1034. fused_reshape_add_reshape_transpose_reshape_transpose22
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[8], 'i'), (nodes[369], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 1035. fused_nn_batch_matmul_51
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[16], 'i'), (nodes[8], 't')]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 1036. fused_reshape_divide_add1
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[8], 'i'), (nodes[7], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 1037. fused_max22
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 1038. fused_subtract_exp1
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[6], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 1039. fused_sum22
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 1040. fused_divide_reshape1
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[16], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 1041. p361
    nodes[370] = addToKV(370, params['p361'])

    # 1042. fused_nn_batch_matmul_393
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[370], 'i'), (nodes[6], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 1043. p362
    nodes[371] = addToKV(371, params['p362'])

    # 1044. fused_reshape_add_reshape_transpose_reshape_transpose_122
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[6], 'i'), (nodes[371], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 1045. fused_nn_batch_matmul_41
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[16], 'i'), (nodes[6], 't')]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 1046. fused_reshape_transpose_reshape1
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[6], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 1047. p363
    nodes[372] = addToKV(372, params['p363'])

    # 1048. fused_nn_batch_matmul_32
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[372], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 1049. p364
    nodes[373] = addToKV(373, params['p364'])

    # 1050. fused_reshape_add_add3
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[16], 'i'), (nodes[373], 'i'), (nodes[13], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 1051. fused_mean45
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a90', output_size, const=False, ephemeral=True))
    arguments = [(nodes[6], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 1052. fused_subtract3
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[6], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 1053. fused_power_mean3
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a91', output_size, const=False, ephemeral=True))
    arguments = [(nodes[8], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 1054. p365
    nodes[374] = addToKV(374, params['p365'])

    # 1055. p366
    nodes[375] = addToKV(375, params['p366'])

    # 1056. fused_add_sqrt_divide_multiply_add2
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[17], 'i'), (nodes[8], 'i'), (nodes[374], 'i'), (nodes[375], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 1057. reshape_nop
    nodes[16] = nodes[16]

    # 1058. p367
    nodes[376] = addToKV(376, params['p367'])

    # 1059. fused_nn_batch_matmul_21
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[376], 'i'), (nodes[13], 't')]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 1060. p368
    nodes[377] = addToKV(377, params['p368'])

    # 1061. fused_reshape_add_multiply_divide_erf_add_multiply_reshape1
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[13], 'i'), (nodes[377], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 1062. p369
    nodes[378] = addToKV(378, params['p369'])

    # 1063. fused_nn_batch_matmul_11
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[378], 'i'), (nodes[8], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 1064. p370
    nodes[379] = addToKV(379, params['p370'])

    # 1065. fused_reshape_add_add2
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[8], 'i'), (nodes[379], 'i'), (nodes[16], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 1066. fused_mean46
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a92', output_size, const=False, ephemeral=True))
    arguments = [(nodes[13], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 1067. fused_subtract2
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[13], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 1068. fused_power_mean2
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a93', output_size, const=False, ephemeral=True))
    arguments = [(nodes[6], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[17], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 1069. p371
    nodes[380] = addToKV(380, params['p371'])

    # 1070. p372
    nodes[381] = addToKV(381, params['p372'])

    # 1071. fused_add_sqrt_divide_multiply_add1
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[17], 'i'), (nodes[6], 'i'), (nodes[380], 'i'), (nodes[381], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 1072. reshape_nop
    nodes[8] = nodes[8]

    # 1073. p373
    nodes[382] = addToKV(382, params['p373'])

    # 1074. fused_nn_batch_matmul_31
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[382], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 1075. p374
    nodes[383] = addToKV(383, params['p374'])

    # 1076. fused_reshape_add_reshape_transpose_reshape
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[16], 'i'), (nodes[383], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 1077. p375
    nodes[384] = addToKV(384, params['p375'])

    # 1078. fused_nn_batch_matmul_394
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[384], 'i'), (nodes[6], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 1079. p376
    nodes[385] = addToKV(385, params['p376'])

    # 1080. fused_reshape_add_reshape_transpose_reshape_transpose23
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[6], 'i'), (nodes[385], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 1081. fused_nn_batch_matmul_5
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[16], 'i'), (nodes[6], 't')]
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 1082. fused_reshape_divide_add
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[6], 'i'), (nodes[7], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 1083. fused_max23
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 1084. fused_subtract_exp
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[13], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 1085. fused_sum23
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[17], 't')]
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 1086. fused_divide_reshape
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[16], 'i'), (nodes[17], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 1087. p377
    nodes[386] = addToKV(386, params['p377'])

    # 1088. fused_nn_batch_matmul_395
    # kernel 0
    arguments = [(nodes[8], 'i'), (nodes[386], 'i'), (nodes[13], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 1089. p378
    nodes[387] = addToKV(387, params['p378'])

    # 1090. fused_reshape_add_reshape_transpose_reshape_transpose_123
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[13], 'i'), (nodes[387], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 1091. fused_nn_batch_matmul_4
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[16], 'i'), (nodes[13], 't')]
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 1092. fused_reshape_transpose_reshape
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[13], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 1093. p379
    nodes[388] = addToKV(388, params['p379'])

    # 1094. fused_nn_batch_matmul_3
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[388], 'i'), (nodes[16], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 1095. p380
    nodes[389] = addToKV(389, params['p380'])

    # 1096. fused_reshape_add_add1
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[16], 'i'), (nodes[389], 'i'), (nodes[8], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 1097. fused_mean47
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a94', output_size, const=False, ephemeral=True))
    arguments = [(nodes[13], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[7], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 1098. fused_subtract1
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[13], 'i'), (nodes[7], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 1099. fused_power_mean1
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a95', output_size, const=False, ephemeral=True))
    arguments = [(nodes[6], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[7], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 1100. p381
    nodes[390] = addToKV(390, params['p381'])

    # 1101. p382
    nodes[391] = addToKV(391, params['p382'])

    # 1102. fused_add_sqrt_divide_multiply_add
    # kernel 0
    arguments = [(nodes[16], 't'), (nodes[7], 'i'), (nodes[6], 'i'), (nodes[390], 'i'), (nodes[391], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 1103. reshape_nop
    nodes[16] = nodes[16]

    # 1104. p383
    nodes[392] = addToKV(392, params['p383'])

    # 1105. fused_nn_batch_matmul_2
    # kernel 0
    arguments = [(nodes[16], 'i'), (nodes[392], 'i'), (nodes[8], 't')]
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 1106. p384
    nodes[393] = addToKV(393, params['p384'])

    # 1107. fused_reshape_add_multiply_divide_erf_add_multiply_reshape
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[8], 'i'), (nodes[393], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 1108. p385
    nodes[394] = addToKV(394, params['p385'])

    # 1109. fused_nn_batch_matmul_1
    # kernel 0
    arguments = [(nodes[13], 'i'), (nodes[394], 'i'), (nodes[6], 't')]
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 1110. p386
    nodes[395] = addToKV(395, params['p386'])

    # 1111. fused_reshape_add_add
    # kernel 0
    arguments = [(nodes[8], 't'), (nodes[6], 'i'), (nodes[395], 'i'), (nodes[16], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 1112. fused_mean48
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a96', output_size, const=False, ephemeral=True))
    arguments = [(nodes[8], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[7], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 1113. fused_subtract
    # kernel 0
    arguments = [(nodes[13], 't'), (nodes[8], 'i'), (nodes[7], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 1114. fused_power_mean
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a97', output_size, const=False, ephemeral=True))
    arguments = [(nodes[13], 'i'), (imm[0], 't')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[7], 't'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 1115. p387
    nodes[396] = addToKV(396, params['p387'])

    # 1116. p388
    nodes[397] = addToKV(397, params['p388'])

    # 1117. fused_add_sqrt_divide_multiply_add_reshape
    # kernel 0
    arguments = [(nodes[6], 't'), (nodes[13], 'i'), (nodes[7], 'i'), (nodes[396], 'i'), (nodes[397], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_reshape_kernel0', path, shapes, arguments))

    # 1118. p389
    nodes[398] = addToKV(398, params['p389'])

    # 1119. fused_nn_batch_matmul
    # kernel 0
    arguments = [(nodes[6], 'i'), (nodes[398], 'i'), (nodes[17], 't')]
    shapes = [(1, 6, 1),  (2, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_kernel0', path, shapes, arguments))

    # 1120. p390
    nodes[399] = addToKV(399, params['p390'])

    # 1121. fused_reshape_add_split
    # kernel 0
    output_size = 1536
    nodes["7_1"] = kaas.bufferSpec('"7_1"', output_size, const=False, ephemeral=True)
    arguments = [(nodes["7_1"], 't'), (nodes[17], 'i'), (nodes[399], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_split_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes["7_0"] = kaas.bufferSpec('"7_0"', output_size, const=False, ephemeral=True)
    arguments = [(nodes["7_0"], 't'), (nodes[17], 'i'), (nodes[399], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_split_kernel1', path, shapes, arguments))

    # 1122. fused_squeeze
    # kernel 0
    output_size = 1536
    nodes[400] = kaas.bufferSpec('400', output_size, const=False, ephemeral=True)
    arguments = [(nodes[400], 'o'), (nodes["7_0"], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_squeeze_kernel0', path, shapes, arguments))

    # 1123. fused_squeeze_1
    # kernel 0
    arguments = [(nodes[17], 'o'), (nodes["7_1"], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_squeeze_1_kernel0', path, shapes, arguments))


    req = kaas.kaasReq(kerns)
    return req

