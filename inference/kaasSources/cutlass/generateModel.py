#!/usr/bin/env python3
import yaml
import pathlib
from cutlassSgemm import createReq
import argparse
import subprocess as sp

cwd = pathlib.Path(__file__).parent.resolve()
modelDir = cwd / ".." / ".." / "models"
cutlassDir = modelDir / "cutlassSgemm"


def getMeta(M, N, K):
    outputs = [{"name": "c", "type": "float32", "shape": [M, N]}]
    inputs = [{"name": "a", "type": "float32", "shape": [M, K]}, {"name": "b", "type": "float32", "shape": [K, N]}]
    return {"constants": [], "inputs": inputs, "outputs": outputs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=pathlib.Path, default=cutlassDir, help="Output Directory")
    parser.add_argument('-n', '--name', default='cutlassSgemm', help="Name to use for output")

    args = parser.parse_args()
    targetDir = args.output
    if not targetDir.exists():
        targetDir.mkdir()

    sp.run(['make'], cwd=cwd, check=True)

    M = 10000
    N = 8000
    K = 10000
    alpha = 1
    beta = 1

    req = createReq(M, N, K, alpha, beta)
    meta_data = getMeta(M, N, K)
    with open(targetDir / (args.name + "_model.yaml"), 'w') as f:
        yaml.safe_dump(req.toDict(), f)

    with open(targetDir / (args.name + "_meta.yaml"), 'w') as f:
        yaml.safe_dump(meta_data, f)
