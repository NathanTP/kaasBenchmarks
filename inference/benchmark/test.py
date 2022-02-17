#!/usr/bin/env python
import argparse
import subprocess as sp
import re
import itertools


# Accuracy for nshot n=32
expectedAccuracies = {
    "resnet50": 0.75,
    'bert': 0.90625,
    'complexCutlassGemm': 1.0
}


def runBench(model, modelType='Kaas', backend='local', experiment='nshot', nRun=1):
    cmd = ["./benchmark.py", "-m", model + modelType, '-b', backend, '-e', experiment, '--numRun', str(nRun)]
    proc = sp.run(cmd, stdout=sp.PIPE, stderr=sp.STDOUT, text=True)
    if proc.returncode != 0:
        print("Command Failed: " + " ".join(cmd))
        print("\n".join(proc.stdout))
        return False

    if "Dataset does not support accuracy calculation" not in proc.stdout:
        match = re.search(r"Accuracy =  (\d\.\d+)", proc.stdout)
        accuracy = float(match.group(1))
        if accuracy != expectedAccuracies[model]:
            print("Unexpected Accuray")
            print("\tExpected: ", expectedAccuracies[model])
            print("\tGot: ", accuracy)
            return False

    return True


def quickTest():
    models = ['resnet50', 'bert', 'complexCutlassGemm', 'jacobi']
    types = ['Kaas', 'Tvm']
    backends = ['local', 'ray']
    configs = itertools.product(models, types, backends)
    for model, modelType, backend in configs:
        if not runBench(model, modelType=modelType, nRun=32, backend=backend):
            print(f"Test Failed: {backend} {model}{modelType}")
            return False
        else:
            print(f"Test Success: {backend} {model}{modelType}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Regression/Correctness Testing for kaasBenchmarks")
    parser.add_argument("-t", "--test", choices=['quick'])
    args = parser.parse_args()

    if args.test == 'quick':
        success = quickTest()

    if success:
        print("Success")
    else:
        print("Failure")
