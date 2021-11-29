#!/usr/bin/env python3
import yaml
import pathlib
# from cutlassSgemm import createReq
import cutlassSgemm as model
import argparse
import subprocess as sp
import numpy as np
import pickle
import shutil

sourceDir = pathlib.Path(__file__).parent.resolve()
modelDir = (sourceDir / ".." / ".." / "models").resolve()
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

    print("Making in: ", sourceDir)
    sp.run(['make'], cwd=sourceDir, check=True)
    print(f"Copying from: {sourceDir} to {modelDir}")
    shutil.copy(sourceDir / 'cutlassAdapters.so', cutlassDir / 'cutlassAdapters.so')
    shutil.copy(sourceDir / 'cutlass.cubin', cutlassDir / 'cutlass.cubin')

    rng = np.random.default_rng(0)
    b = rng.random((model.K, model.N), dtype=np.float32)
    d = rng.random((model.N, model.redDim), dtype=np.float32)

    req = model.createReq(model.M, model.N, model.K, model.alpha, model.beta)
    meta_data = getMeta(model.M, model.N, model.K)
    with open(targetDir / (args.name + "_model.yaml"), 'w') as f:
        yaml.safe_dump(req.toDict(), f)

    with open(targetDir / (args.name + "_meta.yaml"), 'w') as f:
        yaml.safe_dump(meta_data, f)

    with open(targetDir / (args.name + "_params.pkl"), 'wb') as f:
        pickle.dump([b, d], f)
