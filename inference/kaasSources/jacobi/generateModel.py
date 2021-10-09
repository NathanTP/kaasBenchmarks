#!/usr/bin/env python3
import yaml
import pathlib
from jacobi import createReq
import argparse
import subprocess as sp
import numpy as np

cwd = pathlib.Path(__file__).parent.resolve()
modelDir = cwd / ".." / ".." / "models"
jacobiDir = modelDir / "jacobi"


def getMeta(N):
    #constants = [{"name": "b", "type": "float32", "shape": [K, N]}, {"name": "d", "type": "float32", "shape": [N, 1]}]
    constants = []
    inputs = [{"name": "A", "type": "float32", "shape": [N, N]}, {"name": "b", "type": "float32", "shape": [N, 1]}]
    outputs = [{"name": "xnew", "type": "float64", "shape": [N*8, 1]}, {"name": "d", "type": "float64", "shape": [1]}]
    return {"constants": constants, "inputs": inputs, "outputs": outputs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=pathlib.Path, default=jacobiDir, help="Output Directory")
    parser.add_argument('-n', '--name', default='jacobi', help="Name to use for output")

    args = parser.parse_args()
    targetDir = args.output
    if not targetDir.exists():
        targetDir.mkdir()

    sp.run(['make'], cwd=cwd, check=True)

    N = 512

    #rng = np.random.default_rng(0)
    #a = rng.random((M, K), dtype=np.float32)
    #b = rng.random((K, N), dtype=np.float32)
    #c = np.zeros(shape=(M, N), dtype=np.float32)
    #d = rng.random((N, 1), dtype=np.float32)
    #e = np.zeros(shape=(M, 1), dtype=np.float32)

    #b = np.asfortranarray(b)
    #d = np.asfortranarray(d)


    req = createReq()
    meta_data = getMeta(N)
    with open(targetDir / (args.name + "_model.yaml"), 'w') as f:
        yaml.safe_dump(req.toDict(), f)

    with open(targetDir / (args.name + "_meta.yaml"), 'w') as f:
        yaml.safe_dump(meta_data, f)

    #with open(targetDir / (args.name + "_params.pkl"), 'wb') as f:
        #pickle.dump([b, d], f)
