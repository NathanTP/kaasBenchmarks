#!/usr/bin/env python
import pathlib
import subprocess as sp
import sys
import signal
import datetime
import os
import argparse

import util


expRoot = pathlib.Path(__file__).resolve().parent
resultsDir = expRoot / 'results'


def linkLatest(newLatest):
    latestDir = resultsDir / 'latest'
    latestDir.unlink(missing_ok=True)
    latestDir.symlink_to(newLatest)


def launchServer(outDir, nClient, modelType, policy, nGpu=None):
    """Launch the benchmark server. outDir is the directory where experiment
    outputs should go. Returns a Popen object. If nGpu is none, all gpus are
    used, otherwise we restrict the server to nGpu."""
    if modelType == 'Kaas':
        modeArg = '--runner_mode=kaas'
    elif modelType == 'Tvm':
        modeArg = '--runner_mode=actor'
    else:
        raise ValueError("Unrecognized model type: " + modelType)

    env = os.environ
    if nGpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(nGpu)])

    cmd = [expRoot / "benchmark.py",
                     "-t", "server",
                     '-b', 'ray',
                     modeArg,
                     '--runner_policy=' + policy,
                     '--numClient=' + str(nClient)]

    return sp.Popen(cmd, cwd=outDir, stdout=sys.stdout, env=env)


def launchClient(scale, model, name, test, outDir, nIter=1):
    cmd = [(expRoot / "benchmark.py"),
           "-t", test,
           "--numRun=" + str(nIter),
           "-b", "client",
           "--name=" + name,
           "-m", model]

    if scale is not None:
        cmd.append("--scale=" + str(scale))

    return sp.Popen(cmd, stdout=sys.stdout, cwd=outDir)


def mlperfMultiOne(modelNames, modelType, nCpy, scale, prefix, resultsDir):
    if modelType == 'Kaas':
        policy = 'balance'
    elif modelType == 'Tvm':
        policy = 'exclusive'
    else:
        raise ValueError("Unrecognized Model Type: " + modelType)

    runners = {}
    for i in range(nCpy):
        for j, modelName in enumerate(modelNames):
            instanceName = f"{prefix}_{modelName}_{j}_{i}"
            runners[instanceName] = launchClient(
                scale,
                modelName + modelType, instanceName,
                'mlperf', resultsDir)

    server = launchServer(resultsDir, len(runners), modelType, policy)

    failed = []
    for name, runner in runners.items():
        runner.wait()
        if runner.returncode != 0:
            failed.append(name)
    server.send_signal(signal.SIGINT)
    server.wait()

    if len(failed) != 0:
        raise RuntimeError("Some runners failed: ", failed)

    for summaryFile in resultsDir.glob("*_summary.txt"):
        with open(summaryFile, 'r') as f:
            summary = f.read()
        if "INVALID" in summary:
            return False

    return True


def mlperfMulti(modelType, prefix="mlperf_multi", outDir="results"):
    suffix = datetime.datetime.now().strftime("%d%m%y-%H%M%S")
    expResultsDir = outDir / f"multi_{modelType}_{suffix}"
    expResultsDir.mkdir(0o700)
    linkLatest(expResultsDir)

    nCpy = 2

    models = [
        "resnet50",
        "resnet50"
    ]

    prefix = f"{prefix}_{modelType}"

    # Attempt to find a valid scale, starting with "perfect" scaling
    nModel = nCpy * len(models)
    scale = (1 / nModel) * util.getNGpu()

    succeedScale = 0
    failureScale = scale

    # Minimum step size when searching
    step = 0.025

    # Binary Search
    found = False
    while not found:
        print("\n\nAttempting scale: ", scale)
        failure = not mlperfMultiOne(models, modelType, nCpy, scale, prefix, expResultsDir)

        if failure:
            failureScale = scale
        else:
            succeedScale = scale

        if (failureScale - succeedScale) <= step:
            found = True
        else:
            scale = succeedScale + ((failureScale - succeedScale) / 2)

    print("Max achievable scale: ", scale)
    return succeedScale


def mlperfOne(baseModel, modelType, prefix="mlperfOne", outDir="results", findPeak=True):
    if modelType == 'Kaas':
        policy = 'balance'
    elif modelType == 'Tvm':
        policy = 'exclusive'
    else:
        raise ValueError("Unrecognized Model Type: " + modelType)

    model = baseModel + modelType
    if findPeak:
        runner = launchClient(None, model, prefix, 'mlperf', outDir)
        server = launchServer(outDir, 1, modelType, policy, nGpu=1)
    else:
        runner = launchClient(1.0, model, prefix, 'mlperf', outDir)
        server = launchServer(outDir, 1, modelType, policy)

    runner.wait()
    server.send_signal(signal.SIGINT)
    server.wait()

    if runner.returncode != 0:
        raise RuntimeError("Run Failed")


def nShot(baseModel, modelType, nIter=1, prefix="nshotOne", outDir="results"):
    server = launchServer(outDir, 1, modelType, 'balance')

    model = baseModel + modelType
    runner = launchClient(1.0, model, prefix, 'nshot', outDir, nIter=nIter)

    runner.wait()
    server.send_signal(signal.SIGINT)
    server.wait()
    if runner.returncode != 0:
        raise RuntimeError("Run Failed")


# def findParams(model):

if __name__ == "__main__":
    if not resultsDir.exists():
        resultsDir.mkdir(0o700)

    parser = argparse.ArgumentParser("Experiments for the benchmark")
    parser.add_argument("-m", "--model",
                        choices=['bert', 'resnet50', 'superRes'],
                        help="Model to run. Not used in mlperfMulti mode.")
    parser.add_argument("-e", "--experiment",
                        choices=['nshot', 'mlperfOne', 'mlperfMulti'],
                        help="Which experiment to run.")
    parser.add_argument("--findPeak",
                        default=False, action="store_true",
                        help="In mlperfOne mode, find peak performance rather than run a fixed-length experiment.")
    parser.add_argument("-t", "--modelType", default='tvm',
                        choices=['kaas', 'tvm'], help="Which model type to use")

    args = parser.parse_args()

    # internally we concatenate the modelType to a camel-case string so we need
    # to convert the first character to uppercase.
    args.modelType = args.modelType[:1].upper() + args.modelType[1:]

    if args.experiment == 'nshot':
        print("Starting nshot")
        nShot(args.model, args.modelType, outDir=resultsDir, nIter=32)
    elif args.experiment == 'mlperfOne':
        print("Starting mlperfOne")
        mlperfOne(args.model, args.modelType, outDir=resultsDir, findPeak=args.findPeak)
    elif args.experiment == 'mlperfMulti':
        print("Starting mlperfMulti")
        mlperfMulti(args.modelType, outDir=resultsDir)
    else:
        raise ValueError("Invalid experiment: ", args.experiment)

    print("DONE")
