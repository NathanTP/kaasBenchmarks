#!/usr/bin/env python

import yaml
import argparse
import sys
import subprocess as sp
import shutil
import pathlib
import numpy as np
import pickle

import libff.kaas as kaas


srcDir = pathlib.Path(__file__).parent.resolve()

libraryName = "sgemm.cubin"
mmKern = "sgemm"

tile_tb_height = 8
tileN = 16
tileM = (tileN * tile_tb_height)

# This has to match the DIM constant in gemm.cu
sideLength = 128

# Size of one element in bytes, e.g. float32=4
elemSize = 4


def generateLayer(namePrefix, inputName, outputLayer=False, inputLayer=False):
    matSize = (sideLength**2) * elemSize

    aBuf = kaas.bufferSpec(inputName, matSize, ephemeral=(not inputLayer), const=False)
    bBuf = kaas.bufferSpec(namePrefix + "B", matSize, ephemeral=False, const=True)

    outputName = namePrefix + "C"
    cBuf = kaas.bufferSpec(namePrefix + "C", matSize, ephemeral=(not outputLayer), const=False)

    gridDim = (sideLength // tileM, sideLength // tileN, 1)
    blockDim = (tileN, tile_tb_height, 1)
    sharedSize = tile_tb_height * tileN * 4

    arguments = [(aBuf, 'i'),
                 (bBuf, 'i'),
                 (cBuf, 'o')]

    # libraryName will be overwritten by the client when it loads
    # the model, as will the buffer keys
    kern = kaas.kernelSpec(libraryName, mmKern,
                           gridDim, blockDim, sharedSize,
                           literals=[],
                           arguments=arguments)

    return (kern, outputName)


def generateModel(depth):
    layers = []
    layer, previousOut = generateLayer("input", "inputA", outputLayer=False, inputLayer=True)
    layers.append(layer)

    for i in range(depth - 2):
        layer, previousOut = generateLayer("intermediate" + str(i), previousOut,
                                           outputLayer=False,
                                           inputLayer=False)
        layers.append(layer)

    layer, _ = generateLayer("output", previousOut, outputLayer=True, inputLayer=False)
    layers.append(layer)

    return kaas.kaasReq(layers)


def metaFromReq(req):
    shape = (sideLength, sideLength)
    dtype = "float32"

    constants = []
    inputs = []
    outputs = []
    for kern in req.kernels:
        for buf in kern.inputs:
            if not buf.ephemeral:
                if buf.const:
                    constants.append({"name": buf.name, "type": dtype, "shape": shape})
                else:
                    inputs.append({"name": buf.name, "type": dtype, "shape": shape})
        for buf in kern.outputs:
            if not buf.ephemeral:
                outputs.append({"name": buf.name, "type": dtype, "shape": shape})

    return {"constants": constants, "inputs": inputs, "outputs": outputs}


def generateCubin(outputPath):
    sp.run(["make"], cwd="kerns/", check=True)
    shutil.copy("kerns/sgemm.cubin", outputPath)


def generateConstants(depth):
    consts = []
    for i in range(depth):
        const = np.zeros((sideLength, sideLength), dtype=np.float32)
        np.fill_diagonal(const, i+1)
        consts.append(const)
    return consts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--depth', type=int, default=3, help="How many layers of sgemm to generate")
    parser.add_argument('-n', '--name', type=str, default="sgemm", help="Base name for outputs")
    parser.add_argument('-o', '--output', type=pathlib.Path, default=srcDir, help="Output Directory")

    args = parser.parse_args()

    if args.depth < 3:
        print("Depth must be >= 3")
        sys.exit(1)

    if not args.output.exists():
        args.output.mkdir(mode=0o700, parents=True)

    req = generateModel(args.depth)
    with open(args.output / (args.name + "_model.yaml"), 'w') as f:
        yaml.safe_dump(req.toDict(), f)

    meta = metaFromReq(req)
    with open(args.output / (args.name + "_meta.yaml"), 'w') as f:
        yaml.safe_dump(meta, f)

    consts = generateConstants(args.depth)
    with open(args.output / (args.name + "_params.pkl"), 'wb') as f:
        pickle.dump(consts, f)

    generateCubin(args.output / (args.name + ".cubin"))
