#!/usr/bin/env python
import argparse
import subprocess as sp
import re
import itertools


# Accuracy for nshot n=32
expectedAccuracies = {
    "resnet50": 0.75,
    'bert': 0.90625,
    'complexCutlassGemm': 1.0,
    'testModel': 1.0
}


def runBench(model, modelType='kaas', backend='local', experiment='nshot', nRun=1):
    modelType = modelType.capitalize()
    cmd = ["./benchmark.py", "-m", model + modelType, '-b', backend, '-e', experiment, '--numRun', str(nRun)]
    proc = sp.run(cmd, stdout=sp.PIPE, stderr=sp.STDOUT, text=True)
    if proc.returncode != 0:
        print("Command Failed: " + " ".join(cmd))
        print(proc.stdout)
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


def quick():
    models = ['testModel', 'resnet50', 'bert', 'complexCutlassGemm', 'jacobi']
    types = ['kaas', 'tvm']
    backends = ['local', 'ray']
    configs = itertools.product(models, types, backends)
    for model, modelType, backend in configs:
        if not runBench(model, modelType=modelType, nRun=32, backend=backend):
            print(f"Test Failed: {backend} {model}{modelType}")
            return False
        else:
            print(f"Test Success: {backend} {model}{modelType}")
    return True


def runServerMode(model, modelType='kaas', experiment='nshot', n=1, scale=1.0, runTime=None):
    cmd = ["./experiment.py", "-m", model, '-t', modelType, '-e', experiment, '-n', str(n), '-s', str(scale)]
    if runTime is not None:
        cmd += ['--runTime', str(runTime)]

    proc = sp.run(cmd, stdout=sp.PIPE, stderr=sp.STDOUT, text=True)
    if proc.returncode != 0:
        print("Command Failed: " + " ".join(cmd))
        print(proc.stdout)
        return False

    return True


def serverModeQuick():
    models = ['testModel', 'resnet50', 'bert', 'complexCutlassGemm', 'jacobi']
    types = ['kaas', 'tvm']
    configs = itertools.product(models, types)
    for model, modelType in configs:
        if not runServerMode(model, modelType=modelType, n=8):
            print(f"Test Failed: {model}{modelType}")
            return False
        else:
            print(f"Test Success: {model}{modelType}")
    return True


def runMlperf(model, modelType, mode, runTime=30, scale=0.5):
    if mode == 'direct':
        cmd = ["./benchmark.py", "-m", model + modelType.capitalize(), '-b', 'ray',
               '-e', 'mlperf', '--runTime', str(runTime), '--scale', str(scale)]
    elif mode == 'server':
        cmd = ["./experiment.py", "-m", model, '-t', modelType,
               '-e', 'mlperfOne', '--runTime', str(runTime), '--scale', str(scale)]
    else:
        print("UNRECOGNIZED MODE")
        return False

    proc = sp.run(cmd, stdout=sp.PIPE, stderr=sp.STDOUT, text=True)
    if proc.returncode != 0:
        print("Command Failed: " + " ".join(cmd))
        print(proc.stdout)
        return False

    return True


def mlperfQuick():
    models = ['testModel']
    types = ['kaas', 'tvm']
    modes = ['server', 'direct']

    configs = itertools.product(models, types, modes)
    for model, modelType, mode in configs:
        if not runMlperf(model, modelType, mode):
            print(f"Test Failed: {mode} {model}{modelType}")
            return False
        else:
            print(f"Test Success: {mode} {model}{modelType}")

    return True


def smoke():
    models = ['testModel']
    types = ['kaas', 'tvm']
    configs = itertools.product(models, types)
    for model, modelType in configs:
        print(f"\nRunning nshot tests ({modelType}:")
        if not runBench(model, modelType=modelType, experiment='nshot', nRun=32, backend='local'):
            print(f"Test Failed: local {model}{modelType}")
            return False
        else:
            print(f"Test Success: local {model}{modelType}")

        if not runBench(model, modelType=modelType, experiment='nshot', nRun=32, backend='ray'):
            print(f"Test Failed: ray {model}{modelType}")
            return False
        else:
            print(f"Test Success: ray {model}{modelType}")

        print(f"\nRunning server mode test ({modelType}):")
        if not runServerMode(model, modelType=modelType, n=8):
            print(f"Test Failed: {model}{modelType}")
            return False
        else:
            print(f"Test Success: {model}{modelType}")

        print(f"\nRunning mlperf test ({modelType})")
        if not runMlperf(model, modelType, 'direct'):
            print(f"Test Failed: direct {model}{modelType}")
            return False
        else:
            print(f"Test Success: direct {model}{modelType}")

        if not runMlperf(model, modelType, 'server'):
            print(f"Test Failed: server {model}{modelType}")
            return False
        else:
            print(f"Test Success: server {model}{modelType}")

    return True


if __name__ == "__main__":
    availableTests = ['quick', 'serverQuick', 'mlperfQuick']
    parser = argparse.ArgumentParser("Regression/Correctness Testing for kaasBenchmarks")
    parser.add_argument("-t", "--test", action='append', choices=availableTests + ['smoke', 'all'])
    args = parser.parse_args()

    if 'all' in args.test:
        args.test = availableTests

    for test in args.test:
        print("Running: ", test)
        if test == 'smoke':
            success = smoke()
        elif test == 'quick':
            success = quick()
        elif test == 'serverQuick':
            success = serverModeQuick()
        elif test == 'mlperfQuick':
            success = mlperfQuick()
        else:
            raise RuntimeError("Unrecognized Test: ", test)

        print(f"{test} Test Results:")
        if success:
            print("Success\n")
        else:
            print("Failure\n")
