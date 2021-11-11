#!/usr/bin/env python3
import yaml
import pathlib
from jacobi import createReq
import argparse
import subprocess as sp

cwd = pathlib.Path(__file__).parent.resolve()
modelDir = cwd / ".." / ".." / "models"
jacobiDir = modelDir / "jacobi"


def getMeta(N):
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

    req = createReq()
    meta_data = getMeta(N)
    with open(targetDir / (args.name + "_model.yaml"), 'w') as f:
        yaml.safe_dump(req.toDict(), f)

    with open(targetDir / (args.name + "_meta.yaml"), 'w') as f:
        yaml.safe_dump(meta_data, f)
