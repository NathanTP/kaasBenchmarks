#!/usr/bin/env python3
import yaml
import pathlib
from complexCutlassGemm import createReq
import argparse
import subprocess as sp
import numpy as np

cwd = pathlib.Path(__file__).parent.resolve()
modelDir = cwd / ".." / ".." / "models"
cutlassDir = modelDir / "complexCutlassGemm"


def getMeta(M, N, K):
    constants = [{"name": "b", "type": "csingle", "shape": [K, N]}, {"name": "d", "type": "csingle", "shape": [N, 1]}]
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

    M = 100
    N = 8000
    K = 10000
    alpha = 1
    beta = 1

    rng = np.random.default_rng(0)
    a = rng.random((M, K), dtype=np.float32) + rng.random((M, K), dtype=np.float32) * (1j)
    b = rng.random((K, N), dtype=np.float32) + rng.random((K, N), dtype=np.float32) * (1j)
    c = rng.random((M, N), dtype=np.float32) + rng.random((M, N), dtype=np.float32) * (1j)
    d = rng.random((N, 1), dtype=np.float32) + rng.random((N, 1), dtype=np.float32) * (1j)
    e = rng.random((M, 1), dtype=np.float32) + rng.random((M, 1), dtype=np.float32) * (1j)

    req = createReq(M, N, K, alpha, beta, a, b, c, d, e)
    meta_data = getMeta(M, N, K)
    with open(targetDir / (args.name + "_model.yaml"), 'w') as f:
        yaml.safe_dump(req.toDict(), f)

    with open(targetDir / (args.name + "_meta.yaml"), 'w') as f:
        yaml.safe_dump(meta_data, f)
