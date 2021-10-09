#!/usr/bin/env python3
import yaml
import pathlib
from cutlassSgemm import createReq
import argparse
import subprocess as sp
import numpy as np
import pickle

cwd = pathlib.Path(__file__).parent.resolve()
modelDir = cwd / ".." / ".." / "models"
cutlassDir = modelDir / "cutlassSgemm"


def getMeta(M, N, K):
    constants = [{"name": "b", "type": "float32", "shape": [K, N]}, {"name": "d", "type": "float32", "shape": [N, 1]}]
    outputs = [{"name": "e", "type": "float32", "shape": [M, 1]}]
    inputs = [{"name": "a", "type": "float32", "shape": [M, K]}]
    return {"constants": constants, "inputs": inputs, "outputs": outputs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=pathlib.Path, default=cutlassDir, help="Output Directory")
    parser.add_argument('-n', '--name', default='cutlassSgemm', help="Name to use for output")

    args = parser.parse_args()
    targetDir = args.output
    if not targetDir.exists():
        targetDir.mkdir()

    sp.run(['make'], cwd=cwd, check=True)

    M = 100
    N = 8000
    K = 10000
    alpha = 1
    beta = 1

    rng = np.random.default_rng(0)
    a = rng.random((M, K), dtype=np.float32)
    b = rng.random((K, N), dtype=np.float32)
    c = np.zeros(shape=(M, N), dtype=np.float32)
    d = rng.random((N, 1), dtype=np.float32)
    e = np.zeros(shape=(M, 1), dtype=np.float32)

    #b = np.asfortranarray(b)
    #d = np.asfortranarray(d)


    req = createReq(M, N, K, alpha, beta, a, b, c, d, e)
    meta_data = getMeta(M, N, K)
    with open(targetDir / (args.name + "_model.yaml"), 'w') as f:
        yaml.safe_dump(req.toDict(), f)

    with open(targetDir / (args.name + "_meta.yaml"), 'w') as f:
        yaml.safe_dump(meta_data, f)

    with open(targetDir / (args.name + "_params.pkl"), 'wb') as f:
        pickle.dump([b, d], f)
