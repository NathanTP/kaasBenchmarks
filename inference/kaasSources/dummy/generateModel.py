#!/usr/bin/env python
import subprocess as sp
import shutil
import kaas
import argparse
import pathlib
import yaml


srcDir = pathlib.Path(__file__).parent.resolve()


meta = {
    "constants": [],
    "inputs": [{"name": "dummyInput", "type": "uint8", "shape": (1024,)}],
    "outputs": [{"name": "dummyOutput", "type": "uint8", "shape": (1024,)}]
}


def generateCubin(outputPath):
    sp.run(["make"], cwd=srcDir, check=True)
    shutil.copy("dummy.cubin", outputPath)


def generateModel(libraryPath):
    inpBuf = kaas.bufferSpec('dummyInput', 1024, ephemeral=False)
    outBuf = kaas.bufferSpec('dummyOutput', 1024, ephemeral=False)

    kern = kaas.kernelSpec(libraryPath, 'dummy',
                           (1, 1), (1, 1), 0,
                           arguments=[(inpBuf, 'i'), (outBuf, 'o')])

    return kaas.kaasReq([kern])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, default="dummy", help="Base name for outputs")
    parser.add_argument('-o', '--output', type=pathlib.Path, default=srcDir, help="Output Directory")

    args = parser.parse_args()

    libraryPath = args.output / (args.name + ".cubin")

    if not args.output.exists():
        args.output.mkdir(mode=0o700, parents=True)

    req = generateModel(libraryPath)
    with open(args.output / (args.name + "_model.yaml"), 'w') as f:
        yaml.safe_dump(req.toDict(), f)

    with open(args.output / (args.name + "_meta.yaml"), 'w') as f:
        yaml.safe_dump(meta, f)

    generateCubin(libraryPath)
