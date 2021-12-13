#!/usr/bin/env python3
import yaml
import pathlib
from complexCutlassGemm import createReq
from complexCutlassGemm import generateData
import argparse
import subprocess as sp
import shutil
import pickle

cwd = pathlib.Path(__file__).parent.resolve()
modelDir = cwd / ".." / ".." / "models"
cutlassDir = modelDir / "complexCutlassGemm"


def getMeta(M, N, K):
    constants = [{"name": "b", "type": "csingle", "shape": [K, N]},
                 {"name": "d", "type": "csingle", "shape": [N, 1]}]
    outputs = [{"name": "e", "type": "csingle", "shape": [M, 1]}]
    inputs = [{"name": "a", "type": "csingle", "shape": [M, K]}]
    return {"constants": constants, "inputs": inputs, "outputs": outputs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=pathlib.Path, default=cutlassDir, help="Output Directory")
    parser.add_argument('-n', '--name', default='complexCutlassGemm', help="Name to use for output")

    args = parser.parse_args()
    targetDir = args.output
    if not targetDir.exists():
        targetDir.mkdir()

    sp.run(['make'], cwd=cwd, check=True)

    shutil.copy(cwd / 'cutlassAdapters.so', cutlassDir / 'cutlassAdapters.so')
    shutil.copy(cwd / 'cutlass.cubin', cutlassDir / 'cutlass.cubin')

    M = 100
    N = 25000
    K = 10000
    alpha = 1
    beta = 0
    redDim = 1

    req = createReq(M, N, K, redDim, alpha, beta)
    meta_data = getMeta(M, N, K)
    with open(targetDir / (args.name + "_model.yaml"), 'w') as f:
        yaml.safe_dump(req.toDict(), f)

    with open(targetDir / (args.name + "_meta.yaml"), 'w') as f:
        yaml.safe_dump(meta_data, f)

    inp, consts = generateData(M, N, K, redDim)
    with open(targetDir / (args.name + "_input.pkl"), 'wb') as f:
        pickle.dump(inp, f)

    with open(targetDir / (args.name + "_consts.pkl"), 'wb') as f:
        pickle.dump(consts, f)
