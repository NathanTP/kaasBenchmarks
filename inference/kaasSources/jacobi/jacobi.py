import pathlib
import kaas

testPath = pathlib.Path(__file__).resolve().parent


def createReq(mode='direct'):
    N = 512

    # Inputs
    ABuf = kaas.bufferSpec('A', N*N*4, key="A")
    bBuf = kaas.bufferSpec('b', N*8, key="b")

    # Intermediate
    xBuf = kaas.bufferSpec('x', N*8, ephemeral=True)

    # Outputs
    xnewBuf = kaas.bufferSpec('xnew', N*8, ephemeral=True)
    dBuf = kaas.bufferSpec('d', 8, ephemeral=True)

    # Mark xnewBuf and dBuf as temporary from the perspective of kernel 1 (it
    # doesn't output them to the KV store)
    arguments1 = [(ABuf, 'i'), (bBuf, 'i'), (xBuf, 't'), (xnewBuf, 't'), (dBuf, 't')]

    # Kern 2 actually writes xnewBuf and dBuf back so they're marked 'o'
    arguments2 = [(ABuf, 'i'), (bBuf, 'i'), (xnewBuf, 'o'), (xBuf, 't'), (dBuf, 'o')]

    kern1 = kaas.kernelSpec(testPath / 'jacobi.ptx',
                            'JacobiMethod',
                            (256, 1, 1), (66, 1, 1), 8*N,
                            literals=[kaas.literalSpec('i', N)],
                            arguments=arguments1)

    kern2 = kaas.kernelSpec(testPath / 'jacobi.ptx',
                            'JacobiMethod',
                            (256, 1, 1), (66, 1, 1), 8*N,
                            literals=[kaas.literalSpec('i', N)],
                            arguments=arguments2)

    req = kaas.kaasReq([kern1, kern2], nIter=1500)

    return req
