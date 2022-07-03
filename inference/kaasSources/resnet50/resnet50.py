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
    inp = np.zeros((1, 3, 224, 224))
    nodes[0] = addToKV(0, inp, const=False, ephemeral=False)
    # 1. p0
    nodes[1] = addToKV(1, params['p0'])

    # 2. p1
    nodes[2] = addToKV(2, params['p1'])

    # 3. fused_nn_conv2d_add_nn_relu_11
    # kernel 0
    output_size = 3211264
    nodes[3] = kaas.bufferSpec('3', output_size, const=True, ephemeral=True)
    arguments = [(nodes[0], 'i'), (nodes[1], 'i'), (nodes[3], 't'), (nodes[2], 'i')]
    shapes = [(1, 112, 1), (16, 1, 8)]
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_11_kernel0', path, shapes, arguments))

    # 4. fused_nn_max_pool2d
    # kernel 0
    output_size = 3211264
    nodes[4] = kaas.bufferSpec('4', output_size, const=True, ephemeral=True)
    arguments = [(nodes[3], 'i'), (nodes[4], 't')]
    shapes = [(196, 1, 1), (1024, 1, 1)]
    kerns.append(makeKern('fused_nn_max_pool2d_kernel0', path, shapes, arguments))

    # 5. p2
    nodes[5] = addToKV(5, params['p2'])

    # 6. p3
    nodes[6] = addToKV(6, params['p3'])

    # 7. fused_nn_conv2d_add_nn_relu_10
    # kernel 0
    arguments = [(nodes[4], 'i'), (nodes[5], 'i'), (nodes[3], 't'), (nodes[6], 'i')]
    shapes = [(1, 56, 1), (28, 1, 8)]
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_10_kernel0', path, shapes, arguments))

    # 8. p4
    nodes[7] = addToKV(7, params['p4'])

    # 9. p5
    nodes[8] = addToKV(8, params['p5'])

    # 10. fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_32
    imm = []
    # kernel 0
    output_size = 1806336
    imm.append(kaas.bufferSpec('a0', output_size, const=True, ephemeral=True))
    arguments = [(nodes[3], 'i'), (imm[0], 't')]
    shapes = [(98, 1, 1), (128, 1, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_3_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1806336
    imm.append(kaas.bufferSpec('a1', output_size, const=True, ephemeral=True))
    arguments = [(nodes[7], 'i'), (imm[0], 'i'), (imm[1], 't')]
    shapes = [(1, 2, 36), (98, 2, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_3_kernel1', path, shapes, arguments))
    # kernel 2
    output_size = 3211264
    nodes[9] = kaas.bufferSpec('9', output_size, const=True, ephemeral=True)
    arguments = [(imm[1], 'i'), (nodes[9], 't'), (nodes[8], 'i')]
    shapes = [(98, 1, 1), (128, 1, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_3_kernel2', path, shapes, arguments))

    # 11. p6
    nodes[10] = addToKV(10, params['p6'])

    # 12. p7
    nodes[11] = addToKV(11, params['p7'])

    # 13. p8
    nodes[12] = addToKV(12, params['p8'])

    # 14. p9
    nodes[13] = addToKV(13, params['p9'])

    # 15. fused_nn_conv2d_add
    # kernel 0
    arguments = [(nodes[4], 'i'), (nodes[12], 'i'), (nodes[3], 't'), (nodes[13], 'i')]
    shapes = [(2, 14, 4), (28, 1, 16)]
    kerns.append(makeKern('fused_nn_conv2d_add_kernel0', path, shapes, arguments))

    # 16. fused_nn_conv2d_add_add_nn_relu_32
    # kernel 0
    arguments = [(nodes[9], 'i'), (nodes[10], 'i'), (nodes[4], 't'), (nodes[11], 'i'), (nodes[3], 'i')]
    shapes = [(2, 14, 4), (28, 1, 16)]
    kerns.append(makeKern('fused_nn_conv2d_add_add_nn_relu_3_kernel0', path, shapes, arguments))

    # 17. p10
    nodes[14] = addToKV(14, params['p10'])

    # 18. p11
    nodes[15] = addToKV(15, params['p11'])

    # 19. fused_nn_conv2d_add_nn_relu_91
    # kernel 0
    arguments = [(nodes[4], 'i'), (nodes[14], 'i'), (nodes[9], 't'), (nodes[15], 'i')]
    shapes = [(1, 56, 2), (28, 1, 8)]
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_9_kernel0', path, shapes, arguments))

    # 20. p12
    nodes[16] = addToKV(16, params['p12'])

    # 21. p13
    nodes[17] = addToKV(17, params['p13'])

    # 22. fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_31
    imm = []
    # kernel 0
    output_size = 1806336
    imm.append(kaas.bufferSpec('a2', output_size, const=True, ephemeral=True))
    arguments = [(nodes[9], 'i'), (imm[0], 't')]
    shapes = [(98, 1, 1), (128, 1, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_3_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1806336
    imm.append(kaas.bufferSpec('a3', output_size, const=True, ephemeral=True))
    arguments = [(nodes[16], 'i'), (imm[0], 'i'), (imm[1], 't')]
    shapes = [(1, 2, 36), (98, 2, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_3_kernel1', path, shapes, arguments))
    # kernel 2
    arguments = [(imm[1], 'i'), (nodes[3], 't'), (nodes[17], 'i')]
    shapes = [(98, 1, 1), (128, 1, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_3_kernel2', path, shapes, arguments))

    # 23. p14
    nodes[18] = addToKV(18, params['p14'])

    # 24. p15
    nodes[19] = addToKV(19, params['p15'])

    # 25. fused_nn_conv2d_add_add_nn_relu_31
    # kernel 0
    arguments = [(nodes[3], 'i'), (nodes[18], 'i'), (nodes[9], 't'), (nodes[19], 'i'), (nodes[4], 'i')]
    shapes = [(2, 14, 4), (28, 1, 16)]
    kerns.append(makeKern('fused_nn_conv2d_add_add_nn_relu_3_kernel0', path, shapes, arguments))

    # 26. p16
    nodes[20] = addToKV(20, params['p16'])

    # 27. p17
    nodes[21] = addToKV(21, params['p17'])

    # 28. fused_nn_conv2d_add_nn_relu_9
    # kernel 0
    arguments = [(nodes[9], 'i'), (nodes[20], 'i'), (nodes[3], 't'), (nodes[21], 'i')]
    shapes = [(1, 56, 2), (28, 1, 8)]
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_9_kernel0', path, shapes, arguments))

    # 29. p18
    nodes[22] = addToKV(22, params['p18'])

    # 30. p19
    nodes[23] = addToKV(23, params['p19'])

    # 31. fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_3
    imm = []
    # kernel 0
    output_size = 1806336
    imm.append(kaas.bufferSpec('a4', output_size, const=True, ephemeral=True))
    arguments = [(nodes[3], 'i'), (imm[0], 't')]
    shapes = [(98, 1, 1), (128, 1, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_3_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1806336
    imm.append(kaas.bufferSpec('a5', output_size, const=True, ephemeral=True))
    arguments = [(nodes[22], 'i'), (imm[0], 'i'), (imm[1], 't')]
    shapes = [(1, 2, 36), (98, 2, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_3_kernel1', path, shapes, arguments))
    # kernel 2
    arguments = [(imm[1], 'i'), (nodes[4], 't'), (nodes[23], 'i')]
    shapes = [(98, 1, 1), (128, 1, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_3_kernel2', path, shapes, arguments))

    # 32. p20
    nodes[24] = addToKV(24, params['p20'])

    # 33. p21
    nodes[25] = addToKV(25, params['p21'])

    # 34. fused_nn_conv2d_add_add_nn_relu_3
    # kernel 0
    arguments = [(nodes[4], 'i'), (nodes[24], 'i'), (nodes[3], 't'), (nodes[25], 'i'), (nodes[9], 'i')]
    shapes = [(2, 14, 4), (28, 1, 16)]
    kerns.append(makeKern('fused_nn_conv2d_add_add_nn_relu_3_kernel0', path, shapes, arguments))

    # 35. p22
    nodes[26] = addToKV(26, params['p22'])

    # 36. p23
    nodes[27] = addToKV(27, params['p23'])

    # 37. fused_nn_conv2d_add_nn_relu_8
    # kernel 0
    arguments = [(nodes[3], 'i'), (nodes[26], 'i'), (nodes[4], 't'), (nodes[27], 'i')]
    shapes = [(2, 14, 4), (28, 1, 8)]
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_8_kernel0', path, shapes, arguments))

    # 38. p24
    nodes[28] = addToKV(28, params['p24'])

    # 39. p25
    nodes[29] = addToKV(29, params['p25'])

    # 40. fused_nn_conv2d_add_nn_relu_7
    # kernel 0
    arguments = [(nodes[4], 'i'), (nodes[28], 'i'), (nodes[9], 't'), (nodes[29], 'i')]
    shapes = [(1, 14, 2), (14, 1, 16)]
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_7_kernel0', path, shapes, arguments))

    # 41. p26
    nodes[30] = addToKV(30, params['p26'])

    # 42. p27
    nodes[31] = addToKV(31, params['p27'])

    # 43. p28
    nodes[32] = addToKV(32, params['p28'])

    # 44. p29
    nodes[33] = addToKV(33, params['p29'])

    # 45. fused_nn_conv2d_add_1
    # kernel 0
    arguments = [(nodes[3], 'i'), (nodes[32], 'i'), (nodes[4], 't'), (nodes[33], 'i')]
    shapes = [(1, 28, 4), (4, 1, 32)]
    kerns.append(makeKern('fused_nn_conv2d_add_1_kernel0', path, shapes, arguments))

    # 46. fused_nn_conv2d_add_add_nn_relu_23
    # kernel 0
    arguments = [(nodes[9], 'i'), (nodes[30], 'i'), (nodes[3], 't'), (nodes[31], 'i'), (nodes[4], 'i')]
    shapes = [(1, 7, 4), (28, 1, 32)]
    kerns.append(makeKern('fused_nn_conv2d_add_add_nn_relu_2_kernel0', path, shapes, arguments))

    # 47. p30
    nodes[34] = addToKV(34, params['p30'])

    # 48. p31
    nodes[35] = addToKV(35, params['p31'])

    # 49. fused_nn_conv2d_add_nn_relu_62
    # kernel 0
    arguments = [(nodes[3], 'i'), (nodes[34], 'i'), (nodes[9], 't'), (nodes[35], 'i')]
    shapes = [(1, 14, 4), (28, 1, 16)]
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_6_kernel0', path, shapes, arguments))

    # 50. p32
    nodes[36] = addToKV(36, params['p32'])

    # 51. p33
    nodes[37] = addToKV(37, params['p33'])

    # 52. fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_22
    imm = []
    # kernel 0
    output_size = 1806336
    imm.append(kaas.bufferSpec('a6', output_size, const=True, ephemeral=True))
    arguments = [(nodes[9], 'i'), (imm[0], 't')]
    shapes = [(196, 1, 1), (128, 1, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_2_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1806336
    imm.append(kaas.bufferSpec('a7', output_size, const=True, ephemeral=True))
    arguments = [(nodes[36], 'i'), (imm[0], 'i'), (imm[1], 't')]
    shapes = [(2, 4, 16), (49, 4, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_2_kernel1', path, shapes, arguments))
    # kernel 2
    arguments = [(imm[1], 'i'), (nodes[4], 't'), (nodes[37], 'i')]
    shapes = [(196, 1, 1), (128, 1, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_2_kernel2', path, shapes, arguments))

    # 53. p34
    nodes[38] = addToKV(38, params['p34'])

    # 54. p35
    nodes[39] = addToKV(39, params['p35'])

    # 55. fused_nn_conv2d_add_add_nn_relu_22
    # kernel 0
    arguments = [(nodes[4], 'i'), (nodes[38], 'i'), (nodes[9], 't'), (nodes[39], 'i'), (nodes[3], 'i')]
    shapes = [(1, 7, 4), (28, 1, 32)]
    kerns.append(makeKern('fused_nn_conv2d_add_add_nn_relu_2_kernel0', path, shapes, arguments))

    # 56. p36
    nodes[40] = addToKV(40, params['p36'])

    # 57. p37
    nodes[41] = addToKV(41, params['p37'])

    # 58. fused_nn_conv2d_add_nn_relu_61
    # kernel 0
    arguments = [(nodes[9], 'i'), (nodes[40], 'i'), (nodes[4], 't'), (nodes[41], 'i')]
    shapes = [(1, 14, 4), (28, 1, 16)]
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_6_kernel0', path, shapes, arguments))

    # 59. p38
    nodes[42] = addToKV(42, params['p38'])

    # 60. p39
    nodes[43] = addToKV(43, params['p39'])

    # 61. fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_21
    imm = []
    # kernel 0
    output_size = 1806336
    imm.append(kaas.bufferSpec('a8', output_size, const=True, ephemeral=True))
    arguments = [(nodes[4], 'i'), (imm[0], 't')]
    shapes = [(196, 1, 1), (128, 1, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_2_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1806336
    imm.append(kaas.bufferSpec('a9', output_size, const=True, ephemeral=True))
    arguments = [(nodes[42], 'i'), (imm[0], 'i'), (imm[1], 't')]
    shapes = [(2, 4, 16), (49, 4, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_2_kernel1', path, shapes, arguments))
    # kernel 2
    arguments = [(imm[1], 'i'), (nodes[3], 't'), (nodes[43], 'i')]
    shapes = [(196, 1, 1), (128, 1, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_2_kernel2', path, shapes, arguments))

    # 62. p40
    nodes[44] = addToKV(44, params['p40'])

    # 63. p41
    nodes[45] = addToKV(45, params['p41'])

    # 64. fused_nn_conv2d_add_add_nn_relu_21
    # kernel 0
    arguments = [(nodes[3], 'i'), (nodes[44], 'i'), (nodes[4], 't'), (nodes[45], 'i'), (nodes[9], 'i')]
    shapes = [(1, 7, 4), (28, 1, 32)]
    kerns.append(makeKern('fused_nn_conv2d_add_add_nn_relu_2_kernel0', path, shapes, arguments))

    # 65. p42
    nodes[46] = addToKV(46, params['p42'])

    # 66. p43
    nodes[47] = addToKV(47, params['p43'])

    # 67. fused_nn_conv2d_add_nn_relu_6
    # kernel 0
    arguments = [(nodes[4], 'i'), (nodes[46], 'i'), (nodes[3], 't'), (nodes[47], 'i')]
    shapes = [(1, 14, 4), (28, 1, 16)]
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_6_kernel0', path, shapes, arguments))

    # 68. p44
    nodes[48] = addToKV(48, params['p44'])

    # 69. p45
    nodes[49] = addToKV(49, params['p45'])

    # 70. fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_2
    imm = []
    # kernel 0
    output_size = 1806336
    imm.append(kaas.bufferSpec('a10', output_size, const=True, ephemeral=True))
    arguments = [(nodes[3], 'i'), (imm[0], 't')]
    shapes = [(196, 1, 1), (128, 1, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_2_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1806336
    imm.append(kaas.bufferSpec('a11', output_size, const=True, ephemeral=True))
    arguments = [(nodes[48], 'i'), (imm[0], 'i'), (imm[1], 't')]
    shapes = [(2, 4, 16), (49, 4, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_2_kernel1', path, shapes, arguments))
    # kernel 2
    arguments = [(imm[1], 'i'), (nodes[9], 't'), (nodes[49], 'i')]
    shapes = [(196, 1, 1), (128, 1, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_2_kernel2', path, shapes, arguments))

    # 71. p46
    nodes[50] = addToKV(50, params['p46'])

    # 72. p47
    nodes[51] = addToKV(51, params['p47'])

    # 73. fused_nn_conv2d_add_add_nn_relu_2
    # kernel 0
    arguments = [(nodes[9], 'i'), (nodes[50], 'i'), (nodes[3], 't'), (nodes[51], 'i'), (nodes[4], 'i')]
    shapes = [(1, 7, 4), (28, 1, 32)]
    kerns.append(makeKern('fused_nn_conv2d_add_add_nn_relu_2_kernel0', path, shapes, arguments))

    # 74. p48
    nodes[52] = addToKV(52, params['p48'])

    # 75. p49
    nodes[53] = addToKV(53, params['p49'])

    # 76. fused_nn_conv2d_add_nn_relu_5
    # kernel 0
    arguments = [(nodes[3], 'i'), (nodes[52], 'i'), (nodes[9], 't'), (nodes[53], 'i')]
    shapes = [(2, 7, 4), (2, 4, 16)]
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_5_kernel0', path, shapes, arguments))

    # 77. p50
    nodes[54] = addToKV(54, params['p50'])

    # 78. p51
    nodes[55] = addToKV(55, params['p51'])

    # 79. fused_nn_conv2d_add_nn_relu_4
    # kernel 0
    arguments = [(nodes[9], 'i'), (nodes[54], 'i'), (nodes[4], 't'), (nodes[55], 'i')]
    shapes = [(2, 1, 4), (1, 7, 16)]
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_4_kernel0', path, shapes, arguments))

    # 80. p52
    nodes[56] = addToKV(56, params['p52'])

    # 81. p53
    nodes[57] = addToKV(57, params['p53'])

    # 82. p54
    nodes[58] = addToKV(58, params['p54'])

    # 83. p55
    nodes[59] = addToKV(59, params['p55'])

    # 84. fused_nn_conv2d_add_2
    # kernel 0
    arguments = [(nodes[3], 'i'), (nodes[58], 'i'), (nodes[9], 't'), (nodes[59], 'i')]
    shapes = [(1, 7, 32), (14, 1, 8)]
    kerns.append(makeKern('fused_nn_conv2d_add_2_kernel0', path, shapes, arguments))

    # 85. fused_nn_conv2d_add_add_nn_relu_15
    # kernel 0
    arguments = [(nodes[4], 'i'), (nodes[56], 'i'), (nodes[3], 't'), (nodes[57], 'i'), (nodes[9], 'i')]
    shapes = [(1, 7, 16), (7, 2, 16)]
    kerns.append(makeKern('fused_nn_conv2d_add_add_nn_relu_1_kernel0', path, shapes, arguments))

    # 86. p56
    nodes[60] = addToKV(60, params['p56'])

    # 87. p57
    nodes[61] = addToKV(61, params['p57'])

    # 88. fused_nn_conv2d_add_nn_relu_34
    # kernel 0
    arguments = [(nodes[3], 'i'), (nodes[60], 'i'), (nodes[4], 't'), (nodes[61], 'i')]
    shapes = [(1, 7, 8), (14, 2, 16)]
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_3_kernel0', path, shapes, arguments))

    # 89. p58
    nodes[62] = addToKV(62, params['p58'])

    # 90. p59
    nodes[63] = addToKV(63, params['p59'])

    # 91. fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_14
    imm = []
    # kernel 0
    output_size = 1806336
    imm.append(kaas.bufferSpec('a12', output_size, const=True, ephemeral=True))
    arguments = [(nodes[4], 'i'), (imm[0], 't')]
    shapes = [(98, 1, 1), (128, 1, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1806336
    imm.append(kaas.bufferSpec('a13', output_size, const=True, ephemeral=True))
    arguments = [(nodes[62], 'i'), (imm[0], 'i'), (imm[1], 't')]
    shapes = [(1, 16, 16), (49, 1, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1_kernel1', path, shapes, arguments))
    # kernel 2
    arguments = [(imm[1], 'i'), (nodes[9], 't'), (nodes[63], 'i')]
    shapes = [(98, 1, 1), (128, 1, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1_kernel2', path, shapes, arguments))

    # 92. p60
    nodes[64] = addToKV(64, params['p60'])

    # 93. p61
    nodes[65] = addToKV(65, params['p61'])

    # 94. fused_nn_conv2d_add_add_nn_relu_14
    # kernel 0
    arguments = [(nodes[9], 'i'), (nodes[64], 'i'), (nodes[4], 't'), (nodes[65], 'i'), (nodes[3], 'i')]
    shapes = [(1, 7, 16), (7, 2, 16)]
    kerns.append(makeKern('fused_nn_conv2d_add_add_nn_relu_1_kernel0', path, shapes, arguments))

    # 95. p62
    nodes[66] = addToKV(66, params['p62'])

    # 96. p63
    nodes[67] = addToKV(67, params['p63'])

    # 97. fused_nn_conv2d_add_nn_relu_33
    # kernel 0
    arguments = [(nodes[4], 'i'), (nodes[66], 'i'), (nodes[9], 't'), (nodes[67], 'i')]
    shapes = [(1, 7, 8), (14, 2, 16)]
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_3_kernel0', path, shapes, arguments))

    # 98. p64
    nodes[68] = addToKV(68, params['p64'])

    # 99. p65
    nodes[69] = addToKV(69, params['p65'])

    # 100. fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_13
    imm = []
    # kernel 0
    output_size = 1806336
    imm.append(kaas.bufferSpec('a14', output_size, const=True, ephemeral=True))
    arguments = [(nodes[9], 'i'), (imm[0], 't')]
    shapes = [(98, 1, 1), (128, 1, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1806336
    imm.append(kaas.bufferSpec('a15', output_size, const=True, ephemeral=True))
    arguments = [(nodes[68], 'i'), (imm[0], 'i'), (imm[1], 't')]
    shapes = [(1, 16, 16), (49, 1, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1_kernel1', path, shapes, arguments))
    # kernel 2
    arguments = [(imm[1], 'i'), (nodes[3], 't'), (nodes[69], 'i')]
    shapes = [(98, 1, 1), (128, 1, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1_kernel2', path, shapes, arguments))

    # 101. p66
    nodes[70] = addToKV(70, params['p66'])

    # 102. p67
    nodes[71] = addToKV(71, params['p67'])

    # 103. fused_nn_conv2d_add_add_nn_relu_13
    # kernel 0
    arguments = [(nodes[3], 'i'), (nodes[70], 'i'), (nodes[9], 't'), (nodes[71], 'i'), (nodes[4], 'i')]
    shapes = [(1, 7, 16), (7, 2, 16)]
    kerns.append(makeKern('fused_nn_conv2d_add_add_nn_relu_1_kernel0', path, shapes, arguments))

    # 104. p68
    nodes[72] = addToKV(72, params['p68'])

    # 105. p69
    nodes[73] = addToKV(73, params['p69'])

    # 106. fused_nn_conv2d_add_nn_relu_32
    # kernel 0
    arguments = [(nodes[9], 'i'), (nodes[72], 'i'), (nodes[3], 't'), (nodes[73], 'i')]
    shapes = [(1, 7, 8), (14, 2, 16)]
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_3_kernel0', path, shapes, arguments))

    # 107. p70
    nodes[74] = addToKV(74, params['p70'])

    # 108. p71
    nodes[75] = addToKV(75, params['p71'])

    # 109. fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_12
    imm = []
    # kernel 0
    output_size = 1806336
    imm.append(kaas.bufferSpec('a16', output_size, const=True, ephemeral=True))
    arguments = [(nodes[3], 'i'), (imm[0], 't')]
    shapes = [(98, 1, 1), (128, 1, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1806336
    imm.append(kaas.bufferSpec('a17', output_size, const=True, ephemeral=True))
    arguments = [(nodes[74], 'i'), (imm[0], 'i'), (imm[1], 't')]
    shapes = [(1, 16, 16), (49, 1, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1_kernel1', path, shapes, arguments))
    # kernel 2
    arguments = [(imm[1], 'i'), (nodes[4], 't'), (nodes[75], 'i')]
    shapes = [(98, 1, 1), (128, 1, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1_kernel2', path, shapes, arguments))

    # 110. p72
    nodes[76] = addToKV(76, params['p72'])

    # 111. p73
    nodes[77] = addToKV(77, params['p73'])

    # 112. fused_nn_conv2d_add_add_nn_relu_12
    # kernel 0
    arguments = [(nodes[4], 'i'), (nodes[76], 'i'), (nodes[3], 't'), (nodes[77], 'i'), (nodes[9], 'i')]
    shapes = [(1, 7, 16), (7, 2, 16)]
    kerns.append(makeKern('fused_nn_conv2d_add_add_nn_relu_1_kernel0', path, shapes, arguments))

    # 113. p74
    nodes[78] = addToKV(78, params['p74'])

    # 114. p75
    nodes[79] = addToKV(79, params['p75'])

    # 115. fused_nn_conv2d_add_nn_relu_31
    # kernel 0
    arguments = [(nodes[3], 'i'), (nodes[78], 'i'), (nodes[4], 't'), (nodes[79], 'i')]
    shapes = [(1, 7, 8), (14, 2, 16)]
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_3_kernel0', path, shapes, arguments))

    # 116. p76
    nodes[80] = addToKV(80, params['p76'])

    # 117. p77
    nodes[81] = addToKV(81, params['p77'])

    # 118. fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_11
    imm = []
    # kernel 0
    output_size = 1806336
    imm.append(kaas.bufferSpec('a18', output_size, const=True, ephemeral=True))
    arguments = [(nodes[4], 'i'), (imm[0], 't')]
    shapes = [(98, 1, 1), (128, 1, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1806336
    imm.append(kaas.bufferSpec('a19', output_size, const=True, ephemeral=True))
    arguments = [(nodes[80], 'i'), (imm[0], 'i'), (imm[1], 't')]
    shapes = [(1, 16, 16), (49, 1, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1_kernel1', path, shapes, arguments))
    # kernel 2
    arguments = [(imm[1], 'i'), (nodes[9], 't'), (nodes[81], 'i')]
    shapes = [(98, 1, 1), (128, 1, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1_kernel2', path, shapes, arguments))

    # 119. p78
    nodes[82] = addToKV(82, params['p78'])

    # 120. p79
    nodes[83] = addToKV(83, params['p79'])

    # 121. fused_nn_conv2d_add_add_nn_relu_11
    # kernel 0
    arguments = [(nodes[9], 'i'), (nodes[82], 'i'), (nodes[4], 't'), (nodes[83], 'i'), (nodes[3], 'i')]
    shapes = [(1, 7, 16), (7, 2, 16)]
    kerns.append(makeKern('fused_nn_conv2d_add_add_nn_relu_1_kernel0', path, shapes, arguments))

    # 122. p80
    nodes[84] = addToKV(84, params['p80'])

    # 123. p81
    nodes[85] = addToKV(85, params['p81'])

    # 124. fused_nn_conv2d_add_nn_relu_3
    # kernel 0
    arguments = [(nodes[4], 'i'), (nodes[84], 'i'), (nodes[9], 't'), (nodes[85], 'i')]
    shapes = [(1, 7, 8), (14, 2, 16)]
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_3_kernel0', path, shapes, arguments))

    # 125. p82
    nodes[86] = addToKV(86, params['p82'])

    # 126. p83
    nodes[87] = addToKV(87, params['p83'])

    # 127. fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1
    imm = []
    # kernel 0
    output_size = 1806336
    imm.append(kaas.bufferSpec('a20', output_size, const=True, ephemeral=True))
    arguments = [(nodes[9], 'i'), (imm[0], 't')]
    shapes = [(98, 1, 1), (128, 1, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1806336
    imm.append(kaas.bufferSpec('a21', output_size, const=True, ephemeral=True))
    arguments = [(nodes[86], 'i'), (imm[0], 'i'), (imm[1], 't')]
    shapes = [(1, 16, 16), (49, 1, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1_kernel1', path, shapes, arguments))
    # kernel 2
    arguments = [(imm[1], 'i'), (nodes[3], 't'), (nodes[87], 'i')]
    shapes = [(98, 1, 1), (128, 1, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1_kernel2', path, shapes, arguments))

    # 128. p84
    nodes[88] = addToKV(88, params['p84'])

    # 129. p85
    nodes[89] = addToKV(89, params['p85'])

    # 130. fused_nn_conv2d_add_add_nn_relu_1
    # kernel 0
    arguments = [(nodes[3], 'i'), (nodes[88], 'i'), (nodes[9], 't'), (nodes[89], 'i'), (nodes[4], 'i')]
    shapes = [(1, 7, 16), (7, 2, 16)]
    kerns.append(makeKern('fused_nn_conv2d_add_add_nn_relu_1_kernel0', path, shapes, arguments))

    # 131. p86
    nodes[90] = addToKV(90, params['p86'])

    # 132. p87
    nodes[91] = addToKV(91, params['p87'])

    # 133. fused_nn_conv2d_add_nn_relu_2
    # kernel 0
    arguments = [(nodes[9], 'i'), (nodes[90], 'i'), (nodes[3], 't'), (nodes[91], 'i')]
    shapes = [(1, 7, 16), (14, 2, 8)]
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_2_kernel0', path, shapes, arguments))

    # 134. p88
    nodes[92] = addToKV(92, params['p88'])

    # 135. p89
    nodes[93] = addToKV(93, params['p89'])

    # 136. fused_nn_conv2d_add_nn_relu_1
    # kernel 0
    output_size = 100352
    nodes[94] = kaas.bufferSpec('94', output_size, const=True, ephemeral=True)
    arguments = [(nodes[3], 'i'), (nodes[92], 'i'), (nodes[94], 't'), (nodes[93], 'i')]
    shapes = [(1, 1, 8), (1, 7, 32)]
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_1_kernel0', path, shapes, arguments))

    # 137. p90
    nodes[95] = addToKV(95, params['p90'])

    # 138. p91
    nodes[96] = addToKV(96, params['p91'])

    # 139. p92
    nodes[97] = addToKV(97, params['p92'])

    # 140. p93
    nodes[98] = addToKV(98, params['p93'])

    # 141. fused_nn_conv2d_add_3
    # kernel 0
    arguments = [(nodes[9], 'i'), (nodes[97], 'i'), (nodes[4], 't'), (nodes[98], 'i')]
    shapes = [(1, 7, 16), (7, 1, 32)]
    kerns.append(makeKern('fused_nn_conv2d_add_3_kernel0', path, shapes, arguments))

    # 142. fused_nn_conv2d_add_add_nn_relu2
    # kernel 0
    arguments = [(nodes[94], 'i'), (nodes[95], 'i'), (nodes[3], 't'), (nodes[96], 'i'), (nodes[4], 'i')]
    shapes = [(1, 1, 64), (1, 7, 16)]
    kerns.append(makeKern('fused_nn_conv2d_add_add_nn_relu_kernel0', path, shapes, arguments))

    # 143. p94
    nodes[99] = addToKV(99, params['p94'])

    # 144. p95
    nodes[100] = addToKV(100, params['p95'])

    # 145. fused_nn_conv2d_add_nn_relu1
    # kernel 0
    arguments = [(nodes[3], 'i'), (nodes[99], 'i'), (nodes[94], 't'), (nodes[100], 'i')]
    shapes = [(1, 7, 16), (7, 1, 32)]
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_kernel0', path, shapes, arguments))

    # 146. p96
    nodes[101] = addToKV(101, params['p96'])

    # 147. p97
    nodes[102] = addToKV(102, params['p97'])

    # 148. fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu1
    imm = []
    # kernel 0
    output_size = 1806336
    imm.append(kaas.bufferSpec('a22', output_size, const=True, ephemeral=True))
    arguments = [(nodes[94], 'i'), (imm[0], 't')]
    shapes = [(64, 1, 1), (128, 1, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1806336
    imm.append(kaas.bufferSpec('a23', output_size, const=True, ephemeral=True))
    arguments = [(nodes[101], 'i'), (imm[0], 'i'), (imm[1], 't')]
    shapes = [(1, 8, 16), (8, 16, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_kernel1', path, shapes, arguments))
    # kernel 2
    output_size = 100352
    nodes[103] = kaas.bufferSpec('103', output_size, const=True, ephemeral=True)
    arguments = [(imm[1], 'i'), (nodes[103], 't'), (nodes[102], 'i')]
    shapes = [(64, 1, 1), (128, 1, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_kernel2', path, shapes, arguments))

    # 149. p98
    nodes[104] = addToKV(104, params['p98'])

    # 150. p99
    nodes[105] = addToKV(105, params['p99'])

    # 151. fused_nn_conv2d_add_add_nn_relu1
    # kernel 0
    arguments = [(nodes[103], 'i'), (nodes[104], 'i'), (nodes[9], 't'), (nodes[105], 'i'), (nodes[3], 'i')]
    shapes = [(1, 1, 64), (1, 7, 16)]
    kerns.append(makeKern('fused_nn_conv2d_add_add_nn_relu_kernel0', path, shapes, arguments))

    # 152. p100
    nodes[106] = addToKV(106, params['p100'])

    # 153. p101
    nodes[107] = addToKV(107, params['p101'])

    # 154. fused_nn_conv2d_add_nn_relu
    # kernel 0
    arguments = [(nodes[9], 'i'), (nodes[106], 'i'), (nodes[94], 't'), (nodes[107], 'i')]
    shapes = [(1, 7, 16), (7, 1, 32)]
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_kernel0', path, shapes, arguments))

    # 155. p102
    nodes[108] = addToKV(108, params['p102'])

    # 156. p103
    nodes[109] = addToKV(109, params['p103'])

    # 157. fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu
    imm = []
    # kernel 0
    output_size = 1806336
    imm.append(kaas.bufferSpec('a24', output_size, const=True, ephemeral=True))
    arguments = [(nodes[94], 'i'), (imm[0], 't')]
    shapes = [(64, 1, 1), (128, 1, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1806336
    imm.append(kaas.bufferSpec('a25', output_size, const=True, ephemeral=True))
    arguments = [(nodes[108], 'i'), (imm[0], 'i'), (imm[1], 't')]
    shapes = [(1, 8, 16), (8, 16, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_kernel1', path, shapes, arguments))
    # kernel 2
    arguments = [(imm[1], 'i'), (nodes[103], 't'), (nodes[109], 'i')]
    shapes = [(64, 1, 1), (128, 1, 1)]
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_kernel2', path, shapes, arguments))

    # 158. p104
    nodes[110] = addToKV(110, params['p104'])

    # 159. p105
    nodes[111] = addToKV(111, params['p105'])

    # 160. fused_nn_conv2d_add_add_nn_relu
    # kernel 0
    arguments = [(nodes[103], 'i'), (nodes[110], 'i'), (nodes[4], 't'), (nodes[111], 'i'), (nodes[9], 'i')]
    shapes = [(1, 1, 64), (1, 7, 16)]
    kerns.append(makeKern('fused_nn_conv2d_add_add_nn_relu_kernel0', path, shapes, arguments))

    # 161. fused_mean
    imm = []
    # kernel 0
    output_size = 1806336
    imm.append(kaas.bufferSpec('a26', output_size, const=True, ephemeral=True))
    arguments = [(nodes[4], 'i'), (imm[0], 't')]
    shapes = [(64, 1, 1), (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    arguments = [(nodes[94], 't'), (imm[0], 'i')]
    shapes = [(2, 1, 1), (1024, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 162. fused_squeeze_nn_batch_flatten
    # kernel 0
    arguments = [(nodes[103], 't'), (nodes[94], 'i')]
    shapes = [(2, 1, 1), (1024, 1, 1)]
    kerns.append(makeKern('fused_squeeze_nn_batch_flatten_kernel0', path, shapes, arguments))

    # 163. p106
    nodes[112] = addToKV(112, params['p106'])

    # 164. p107
    nodes[113] = addToKV(113, params['p107'])

    # 165. fused_nn_dense_add
    # kernel 0
    output_size = 4004
    nodes[114] = kaas.bufferSpec('114', output_size, const=True, ephemeral=True)
    arguments = [(nodes[103], 'i'), (nodes[112], 'i'), (nodes[114], 't'), (nodes[113], 'i')]
    shapes = [(1001, 1, 1), (64, 1, 1)]
    kerns.append(makeKern('fused_nn_dense_add_kernel0', path, shapes, arguments))

    # 166. fused_argmax
    # kernel 0
    output_size = 4
    nodes[115] = kaas.bufferSpec('115', output_size, const=True, ephemeral=True)
    arguments = [(nodes[114], 'i'), (nodes[115], 't')]
    shapes = [(1, 1, 1), (32, 32, 1)]
    kerns.append(makeKern('fused_argmax_kernel0', path, shapes, arguments))

    # 167. fused_cast
    # kernel 0
    output_size = 8
    nodes[116] = kaas.bufferSpec('116', output_size, const=True, ephemeral=True)
    arguments = [(nodes[116], 'o'), (nodes[115], 'i')]
    shapes = [(1, 1, 1), (1, 1, 1)]
    kerns.append(makeKern('fused_cast_kernel0', path, shapes, arguments))

    # 168. fused_max
    # kernel 0
    arguments = [(nodes[114], 'i'), (nodes[115], 't')]
    shapes = [(1, 1, 1), (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 169. fused_subtract_exp
    # kernel 0
    output_size = 4004
    nodes[117] = kaas.bufferSpec('117', output_size, const=True, ephemeral=True)
    arguments = [(nodes[117], 't'), (nodes[114], 'i'), (nodes[115], 'i')]
    shapes = [(1, 1, 1), (1001, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 170. fused_sum
    # kernel 0
    arguments = [(nodes[117], 'i'), (nodes[115], 't')]
    shapes = [(1, 1, 1), (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 171. fused_divide
    # kernel 0
    arguments = [(nodes[114], 'o'), (nodes[117], 'i'), (nodes[115], 'i')]
    shapes = [(1, 1, 1), (1001, 1, 1)]
    kerns.append(makeKern('fused_divide_kernel0', path, shapes, arguments))


    req = kaas.kaasReq(kerns)
    return req

