#!/usr/bin/env python
import pathlib
import subprocess as sp
import sys
import signal
import datetime
import os
import argparse
import json
from pprint import pprint
import time
import tempfile
import shutil

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
                     "-e", "server",
                     '-b', 'ray',
                     '--policy=' + policy,
                     '--numClient=' + str(nClient)]

    return sp.Popen(cmd, cwd=outDir, stdout=sys.stdout, env=env)


def launchClient(scale, model, name, test, outDir, nRun=1):
    cmd = [(expRoot / "benchmark.py"),
           "-e", test,
           "--numRun=" + str(nRun),
           "-b", "client",
           "--name=" + name,
           "-m", model]

    if scale is not None:
        cmd.append("--scale=" + str(scale))

    return sp.Popen(cmd, stdout=sys.stdout, cwd=outDir)


def runTest(test, modelNames, modelType, prefix, resultsDir, nCpy=1, scale=1.0, nRun=1):
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
                scale, modelName + modelType, instanceName,
                test, resultsDir, nRun=nRun)

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
                    # raise RuntimeError("Test didn't meet minimum queries, try again with a longer runtime")
                    print("WARNING: Test didn't meet minimum queries, try again with a longer runtime")
                if "Min duration satisfied : NO" in summary:
                    # raise RuntimeError("Test didn't meet minimum duration, try again with a longer runtime")
                    print("WARNING: Test didn't meet minimum duration, try again with a longer runtime")

                return False

    if test == 'nshot':
        modelThroughputs = {name: 0 for name in modelNames}
        for i in range(nCpy):
            for j, modelName in enumerate(modelNames):
                instanceName = f"{prefix}_{modelName}_{j}_{i}"

                with open(resultsDir / (instanceName + "_results.json"), 'r') as f:
                    instanceMetrics = json.load(f)

                modelThroughputs[modelName] += instanceMetrics[0]['metrics']['throughput']

        print("Total throughput: ", modelThroughputs)

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


def mlperfMulti(modelType, prefix="mlperf_multi", outDir="results", scale=None, nCpy=1, model=None):
    suffix = datetime.datetime.now().strftime("%d%m%y-%H%M%S")
    expResultsDir = outDir / f"mlperf_{modelType}_{suffix}"

    # We currently only use homogenous workloads, but we can also make model a
    # list or just manually override if we want to mix models
    models = [model]

    prefix = f"{prefix}_{modelType}"

    # Attempt to find a valid scale, starting with "perfect" scaling
    nModel = nCpy * len(models)
    if scale is None:
        scale = ((1 / nModel) * util.getNGpu())
        startScale = scale
        succeedScale = 0
        failureScale = scale
        runOnce = False
    else:
        # This tricks the system into only running one iteration
        startScale = float('inf')
        succeedScale = scale
        failureScale = scale
        runOnce = True

    # Minimum step size when searching
    step = 0.05

    # Binary Search
    found = False
    while not found:
        print("\n\nAttempting scale: ", scale)
        time.sleep(10)  # ray is bad at cleaning up, gotta wait to be sure
        with tempfile.TemporaryDirectory() as tmpRes:
            tmpRes = pathlib.Path(tmpRes)
            failure = not runTest('mlperf', models, modelType, prefix, tmpRes, nCpy=nCpy, scale=scale)
            if failure:
                failureScale = scale
            else:
                if expResultsDir.exists():
                    shutil.rmtree(expResultsDir)
                shutil.copytree(tmpRes, expResultsDir, ignore=shutil.ignore_patterns("*.ipc"))
                succeedScale = scale

            if (failureScale - succeedScale) <= step:
                # Sometimes we guess wrong and start too low, this bumps us up a
                # bit to make sure we get a valid answer.
                if scale == startScale and not runOnce:
                    scale *= 1.5
                    startScale = scale
                else:
                    found = True
                    # If we never found a passing result, we just report the last
                    # one that ran
                    if not expResultsDir.exists():
                        shutil.copytree(tmpRes, expResultsDir, ignore=shutil.ignore_patterns("*.ipc"))

            else:
                scale = succeedScale + ((failureScale - succeedScale) / 2)

    linkLatest(expResultsDir)

    print("Max achievable scale: ", scale)
    return succeedScale


def mlperfOne(baseModel, modelType, prefix="mlperfOne", outDir="results", scale=None):
    if modelType == 'Kaas':
        policy = 'balance'
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

    runTest('nshot', models, modelType, prefix, expResultsDir, nRun=n)


def nShot(baseModel, modelType, nIter=1, prefix="nshotOne", outDir="results"):
    server = launchServer(outDir, 1, modelType, 'balance')

    model = baseModel + modelType
    runner = launchClient(1.0, model, prefix, 'nshot', outDir, nRun=nIter)

    runner.wait()
    server.send_signal(signal.SIGINT)
    server.wait()
    if runner.returncode != 0:
        raise RuntimeError("Run Failed")


def throughput(modelType, scale=1.0, prefix="throughput", outDir="results", nCpy=1, model=None):
    suffix = datetime.datetime.now().strftime("%d%m%y-%H%M%S")
    expResultsDir = outDir / f"throughput_{modelType}_{suffix}"
    expResultsDir.mkdir(0o700)
    linkLatest(expResultsDir)

    models = [model]*nCpy

    if scale is None:
        scale = ((1 / len(models)) * util.getNGpu())

    prefix = f"{prefix}_{modelType}"

    runTest('throughput', models, modelType, prefix, expResultsDir, scale=scale)

    # Total throughput is queryMS / S. That is, number of ms of actual query
    # work done per second. Another way of thinking of this is QPS normalized
    # by workload size. This is based on the median unloaded query latency
    # rather than direct observation to include any additional overheads the
    # workload might introduce.
    results = {'normalizedThroughput': 0}
    for resultsFile in expResultsDir.glob("*_results.json"):
        with open(resultsFile, 'r') as f:
            result = json.load(f)[0]

        results[result['config']['name']] = result['metrics']['throughput']

        modelSpec = util.getModelSpec(result['config']['model'])
        maxQps, medianLatency = modelSpec.modelClass.getPerfEstimates(util.getGpuType())
        results['normalizedThroughput'] += result['metrics']['throughput'] * medianLatency

    print("Aggregated Results:")
    pprint(results)


if __name__ == "__main__":
    if not resultsDir.exists():
        resultsDir.mkdir(0o700)

    parser = argparse.ArgumentParser("Experiments for the benchmark")
    parser.add_argument("-m", "--model",
                        choices=['bert', 'resnet50', 'superRes', 'complexCutlassGemm'],
                        help="Model to run. Not used in mlperfMulti mode.")
    parser.add_argument("-e", "--experiment",
                        choices=['nshot', 'nshotMulti', 'mlperfOne', 'mlperfMulti', 'throughput'],
                        help="Which experiment to run.")
    parser.add_argument("-t", "--modelType", default='tvm',
                        choices=['kaas', 'tvm'], help="Which model type to use")
    parser.add_argument("-s", "--scale", type=float, help="For mlperf modes, what scale to run each client at. If omitted, tests will try to find peak performance.")
    parser.add_argument("-n", "--nCopy", type=int, help="For mlperfMulti, this is the number of model replicas to use. For nshot, this is the number of iterations.")

    args = parser.parse_args()

    # internally we concatenate the modelType to a camel-case string so we need
    # to convert the first character to uppercase.
    args.modelType = args.modelType[:1].upper() + args.modelType[1:]

    if args.experiment == 'nshot':
        print("Starting nshot")
        if args.nCopy is None:
            nIter = 32
        else:
            nIter = args.nCopy
        nShot(args.model, args.modelType, outDir=resultsDir, nIter=nIter)
    elif args.experiment == 'nshotMulti':
        nShotMulti(32, args.modelType, outDir=resultsDir)
    elif args.experiment == 'mlperfOne':
        print("Starting mlperfOne")
        mlperfOne(args.model, args.modelType, outDir=resultsDir, scale=args.scale)
    elif args.experiment == 'mlperfMulti':
        print("Starting mlperfMulti")
        mlperfMulti(args.modelType, outDir=resultsDir, scale=args.scale, model=args.model, nCpy=args.nCopy)
    elif args.experiment == 'throughput':
        print("Starting Throughput Test")
        throughput(args.modelType, outDir=resultsDir, scale=args.scale, model=args.model, nCpy=args.nCopy)
    else:
        raise ValueError("Invalid experiment: ", args.experiment)

    print("DONE")
