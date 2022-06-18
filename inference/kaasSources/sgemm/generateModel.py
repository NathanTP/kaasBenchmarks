#!/usr/bin/env python

import yaml
import argparse
import sys
import subprocess as sp
import shutil
import pathlib
import numpy as np
import pickle
import math

import kaas

import pycuda.driver as cuda
import pycuda.autoinit  # NOQA

srcDir = pathlib.Path(__file__).parent.resolve()

libraryName = "sgemm.cubin"
mmKern = "sgemm"

sideLen = 1024

maxThreads = cuda.Device(0).get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK)
tileSize = int(math.sqrt(maxThreads))
gridDim = (sideLen // tileSize, sideLen // tileSize, 1)
blockDim = (tileSize, tileSize, 1)
sharedSize = 0

# Size of one element in bytes, e.g. float32=4
elemSize = 4


def generateLayer(namePrefix, inputName, libraryPath, layerIdx, outputLayer=False, inputLayer=False):
    matSize = (sideLen**2) * elemSize

    aBuf = kaas.bufferSpec(inputName, matSize, ephemeral=(not inputLayer), const=False)
    bBuf = kaas.bufferSpec(namePrefix + "_B", matSize, offset=layerIdx * matSize, ephemeral=False, const=True)

    outputName = namePrefix + "_C"
    cBuf = kaas.bufferSpec(namePrefix + "_C", matSize, ephemeral=True, const=False)

    arguments = [(aBuf, 'i'),
                 (bBuf, 'i')]

    if outputLayer:
        arguments.append((cBuf, 'o'))
    else:
        arguments.append((cBuf, 't'))

    # libraryName will be overwritten by the client when it loads
    # the model, as will the buffer keys
    kern = kaas.kernelSpec(libraryPath, mmKern,
                           gridDim, blockDim, sharedSize,
                           literals=[kaas.literalSpec('i', sideLen)],
                           arguments=arguments)

    return (kern, outputName)


def generateModel(depth, libraryPath):
    layers = []
    layer, previousOut = generateLayer("layer0", "layer0_A", libraryPath, 0,
                                       outputLayer=False, inputLayer=True)
    layers.append(layer)

    for i in range(depth - 2):
        layer, previousOut = generateLayer(f"layer{i + 1}", previousOut, libraryPath, i + 1,
                                           outputLayer=False, inputLayer=False)
        layers.append(layer)

    layer, _ = generateLayer(f"layer{depth - 1}", previousOut, libraryPath, depth - 1,
                             outputLayer=True, inputLayer=False)
    layers.append(layer)

    return kaas.kaasReq(layers)


def metaFromReq(req):
    shape = (sideLen, sideLen)
    dtype = "float32"

    constants = []
    inputs = []
    outputs = []
    for kern in req.kernels:
        for bufName, ioType in zip(kern.arguments, kern.ioTypes):
            buf = req.bufferMap[bufName]
            if not buf.ephemeral:
                if buf.const:
                    # XXX Shape is wrong here, gotta think about why it's really here and what to do with it...
                    constants.append({"name": buf.name, "type": dtype, "shape": shape, "dataIdx": 0})
                elif ioType == 'i':
                    inputs.append({"name": buf.name, "type": dtype, "shape": shape})
                elif ioType == 'o':
                    outputs.append({"name": buf.name, "type": dtype, "shape": shape})
                else:
                    raise RuntimeError("Did not expect an io buffer: ", bufName)

    return {"constants": constants, "inputs": inputs, "outputs": outputs}


def generateCubin(outputPath):
    sp.run(["make"], cwd="kerns/", check=True)
    shutil.copy("kerns/sgemm.cubin", outputPath)


def generateConstants(depth):
    consts = []
    # rng = np.random.default_rng(1)
    for i in range(depth):
        const = np.zeros((sideLen, sideLen), dtype=np.float32)
        np.fill_diagonal(const, i+1)
        consts.append(const.flatten())
        # consts.append(rng.standard_normal((sideLen**2), dtype=np.float32))

    combinedConst = np.concatenate(consts)
    return [combinedConst]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--depth', type=int, default=3, help="How many layers of sgemm to generate")
    parser.add_argument('-n', '--name', type=str, default="sgemm", help="Base name for outputs")
    parser.add_argument('-o', '--output', type=pathlib.Path, default=srcDir, help="Output Directory")

    args = parser.parse_args()

    libraryPath = args.output / (args.name + ".cubin")

    if args.depth < 3:
        print("Depth must be >= 3")
        sys.exit(1)

    if not args.output.exists():
        args.output.mkdir(mode=0o700, parents=True)

    req = generateModel(args.depth, libraryPath)
    with open(args.output / (args.name + "_model.pkl"), 'wb') as f:
        pickle.dump(req, f)

    meta = metaFromReq(req)
    with open(args.output / (args.name + "_meta.yaml"), 'w') as f:
        yaml.safe_dump(meta, f)

    consts = generateConstants(args.depth)
    with open(args.output / (args.name + "_params.pkl"), 'wb') as f:
        pickle.dump(consts, f)

    generateCubin(libraryPath)
