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
    env = os.environ
    if nGpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(nGpu)])

    cmd = [expRoot / "benchmark.py",
                     "-t", "server",
                     '-b', 'ray',
                     '--policy=' + policy,
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


def runTest(test, modelNames, modelType, prefix, resultsDir, nCpy=1, scale=1.0):
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
                test, resultsDir)

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

    if test == 'mlperf':
        for summaryFile in resultsDir.glob("*_summary.txt"):
            with open(summaryFile, 'r') as f:
                summary = f.read()
            if "INVALID" in summary:
                if "Min queries satisfied : NO" in summary:
                    raise RuntimeError("Test didn't meet minimum queries, try again with a longer runtime")
                if "Min duration satisfied : NO" in summary:
                    raise RuntimeError("Test didn't meet minimum duration, try again with a longer runtime")

                return False

    return True


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
            if "Min queries satisfied : NO" in summary:
                raise RuntimeError("Test didn't meet minimum queries, try again with a longer runtime")
            if "Min duration satisfied : NO" in summary:
                raise RuntimeError("Test didn't meet minimum duration, try again with a longer runtime")

            return False

    return True


def mlperfMulti(modelType, prefix="mlperf_multi", outDir="results", scale=None):
    suffix = datetime.datetime.now().strftime("%d%m%y-%H%M%S")
    expResultsDir = outDir / f"multi_{modelType}_{suffix}"
    expResultsDir.mkdir(0o700)
    linkLatest(expResultsDir)

    nCpy = 1

    models = [
        "resnet50",
        "resnet50"
    ]

    prefix = f"{prefix}_{modelType}"

    # Attempt to find a valid scale, starting with "perfect" scaling
    nModel = nCpy * len(models)
    if scale is None:
        scale = ((1 / nModel) * util.getNGpu())
        succeedScale = 0
        failureScale = scale
    else:
        # This tricks the system into only running one iteration
        succeedScale = scale
        failureScale = scale

    # Minimum step size when searching
    step = 0.025

    # Binary Search
    found = False
    while not found:
        print("\n\nAttempting scale: ", scale)
        failure = not runTest('mlperf', models, modelType, prefix, expResultsDir, nCpy=nCpy, scale=scale)

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


def mlperfOne(baseModel, modelType, prefix="mlperfOne", outDir="results", scale=None):
    if modelType == 'Kaas':
        policy = 'affinity'
    elif modelType == 'Tvm':
        policy = 'exclusive'
    else:
        raise ValueError("Unrecognized Model Type: " + modelType)

    model = baseModel + modelType
    if scale is None:
        runner = launchClient(None, model, prefix, 'mlperf', outDir)
        server = launchServer(outDir, 1, modelType, policy, nGpu=1)
    else:
        runner = launchClient(scale, model, prefix, 'mlperf', outDir)
        server = launchServer(outDir, 1, modelType, policy)

    runner.wait()
    server.send_signal(signal.SIGINT)
    server.wait()

    if runner.returncode != 0:
        raise RuntimeError("Run Failed")


def nShotMulti(n, modelType, prefix="nshot_multi", outDir="results"):
    suffix = datetime.datetime.now().strftime("%d%m%y-%H%M%S")
    expResultsDir = outDir / f"multi_{modelType}_{suffix}"
    expResultsDir.mkdir(0o700)
    linkLatest(expResultsDir)

    models = [
        "resnet50",
        "resnet50"
    ]

    prefix = f"{prefix}_{modelType}"

    runTest('nshot', models, modelType, prefix, expResultsDir)


def nShot(baseModel, modelType, nIter=1, prefix="nshotOne", outDir="results"):
    server = launchServer(outDir, 1, modelType, 'balance')

    model = baseModel + modelType
    runner = launchClient(1.0, model, prefix, 'nshot', outDir, nIter=nIter)

    runner.wait()
    server.send_signal(signal.SIGINT)
    server.wait()
    if runner.returncode != 0:
        raise RuntimeError("Run Failed")


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
    parser.add_argument("-t", "--modelType", default='tvm',
                        choices=['kaas', 'tvm'], help="Which model type to use")
    parser.add_argument("-s", "--scale", type=float, help="For mlperf modes, what scale to run each client at. If omitted, tests will try to find peak performance.")

    args = parser.parse_args()

    # internally we concatenate the modelType to a camel-case string so we need
    # to convert the first character to uppercase.
    args.modelType = args.modelType[:1].upper() + args.modelType[1:]

    if args.experiment == 'nshot':
        print("Starting nshot")
        nShot(args.model, args.modelType, outDir=resultsDir, nIter=32)
    elif args.experiment == 'mlperfOne':
        print("Starting mlperfOne")
        mlperfOne(args.model, args.modelType, outDir=resultsDir, scale=args.scale)
    elif args.experiment == 'mlperfMulti':
        print("Starting mlperfMulti")
        mlperfMulti(args.modelType, outDir=resultsDir, scale=args.scale)
    else:
        raise ValueError("Invalid experiment: ", args.experiment)

    print("DONE")
