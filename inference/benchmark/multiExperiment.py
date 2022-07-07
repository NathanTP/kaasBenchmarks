#!/usr/bin/env python
import subprocess as sp
import pathlib
import datetime
import shutil
import time
import itertools
import argparse
import numpy as np

from infbench import properties

resultsDir = pathlib.Path("./results")

# nReplicas = [2, 4, 6, 8]
# models = ['cGEMM', 'jacobi', 'resnet50', 'bert']
# expKeys = ['exclusive']


def keyToOpts(expKey):
    """Return arguments to add to the base commands for this mode. These
    include type, policy, and fractional"""
    if expKey == 'kaas':
        return ['-t', 'kaas', '-p', 'affinity']
    elif expKey == 'exclusive':
        return ['-t', 'native', '-p', 'exclusive']
    elif expKey == 'static':
        return ['-t', 'native', '-p', 'static']
    elif expKey == 'fractional':
        return ['-t', 'native', '-p', 'static', '--fractional', 'mem']
    else:
        raise ValueError("Unrecognized experiment: " + expKey)


def getTargetRuntime(nReplica, model, expKey, fast=False):
    if fast:
        return 120

    # After 4 replicas, TVM is so slow that we need lots of time to get a good
    # measurement. How much longer we need depends on the model, though longer
    # is always better.
    if nReplica > 4 and expKey == 'exclusive':
        if model == 'bert':
            runTime = 600
        elif model == 'jacobi':
            runTime = 600
        elif model == 'resnet50':
            runTime = 600
        elif model == 'cGEMM':
            runTime = 600
        else:
            raise RuntimeError("Please configure a target runtime for model: ", model)
    else:
        runTime = 300

    return runTime


def mlperf(configs, suiteOutDir, fast=False):
    if fast:
        scaleArg = ['-s', '0.1']
    else:
        scaleArg = []

    for model, expKey, nReplica in configs:
        runTime = getTargetRuntime(nReplica, model, expKey, fast=fast)

        name = f"{model}_{expKey}_{nReplica}"
        print("\nStarting test: ", name)
        cmd = ['./experiment.py', '-e', 'mlperf',
               '-n', str(nReplica), f'--runTime={runTime}',
               '-m', model]
        cmd += scaleArg
        cmd += keyToOpts(expKey)
        sp.run(cmd)

        runOutDir = suiteOutDir / name
        shutil.copytree(resultsDir / 'latest', runOutDir, ignore=shutil.ignore_patterns("*.ipc"))
        time.sleep(20)

    print("Final Results in: ", suiteOutDir)


def nShot(configs, suiteOutDir, fast=False):
    for model, expKey, nReplica in configs:
        name = f"{model}_{expKey}_{nReplica}"
        print("\nStarting test: ", name)
        cmd = ['./experiment.py',
               '-e', 'nshot',
               '-n', str(nReplica),
               '-m', model]

        cmd += keyToOpts(expKey)

        if fast:
            cmd += ['-s', '1']
        else:
            cmd += ['-s', '64']

        sp.run(cmd)

        runOutDir = suiteOutDir / name
        shutil.copytree(resultsDir / 'latest', runOutDir, ignore=shutil.ignore_patterns("*.ipc"))
        time.sleep(10)

    print("Final Results in: ", suiteOutDir)


def throughput(configs, suiteOutDir, fast=False):
    for model, expKey, nReplica in configs:
        name = f"{model}_{expKey}_{nReplica}"
        runTime = getTargetRuntime(nReplica, model, expKey, fast=fast)

        cmd = ['./experiment.py',
               '-e', 'throughput',
               '-n', str(nReplica),
               '-s', str(1 / nReplica),
               f'--runTime={runTime}',
               '-m', model]
        cmd += keyToOpts(expKey)

        print("\nStarting test: ", name)
        sp.run(cmd)

        runOutDir = suiteOutDir / name
        shutil.copytree(resultsDir / 'latest', runOutDir, ignore=shutil.ignore_patterns("*.ipc"))
        time.sleep(20)

    print("Final Results in: ", suiteOutDir)


def getScales(props, model, expKey, nReplica, fast, hetero=False):
    # How many qps the system can sustain with nReplica
    peakThr = props.throughputFull(model, nReplica, expKey, independent=True)

    if fast:
        safeThr = 0.2 * peakThr
    else:
        safeThr = 0.7 * peakThr

    if hetero:
        zipfFactor = 1.8
        # Find a zipf series that isn't too skewed.
        # Use the same seed for every experiment for fairness
        rng = np.random.default_rng(0)
        raw = rng.zipf(zipfFactor, nReplica)
        while (max(raw) / min(raw)) > 20:
            raw = rng.zipf(zipfFactor, nReplica)
        # We scale the submission rate so that the heaviest model submits at
        # full (safe) throughput. Everyone else is scaled down accordingly.
        heavyScale = peakThr / nReplica
        heavyRaw = max(raw)

        # Normalize the raw distribution to the target scale
        scaleFactor = heavyScale / heavyRaw
        return scaleFactor * raw
    else:
        return [safeThr / nReplica]*nReplica


def latDistribution(configs, suiteOutDir, fast=False, hetero=False):
    props = properties.getProperties()
    for model, expKey, nReplica in configs:
        scales = getScales(props, model, expKey, nReplica, fast, hetero)
        runTime = getTargetRuntime(nReplica, model, expKey, fast=fast)

        name = f"{model}_{expKey}_{nReplica}"
        print("\nStarting test: ", name)
        cmd = ['./experiment.py', '-e', 'mlperf', '-n', str(nReplica),
               f'--runTime={runTime}', '-m', model]
        cmd += ['-s', ",".join(map(str, scales))]
        cmd += keyToOpts(expKey)

        print("Running: ", " ".join(cmd))
        sp.run(cmd)

        runOutDir = suiteOutDir / name
        shutil.copytree(resultsDir / 'latest', runOutDir, ignore=shutil.ignore_patterns("*.ipc"))
        time.sleep(20)

    print("Final Results in: ", suiteOutDir)


def latThr(configs, suiteOutDir, fast=False, hetero=False):
    props = properties.getProperties()
    # sweepScales = [0.2, 0.4, 0.6, 0.8]
    #XXX
    sweepScales = [0.2, 0.4]
    for model, expKey, nReplica in configs:
        scales = getScales(props, model, expKey, nReplica, False, hetero)
        runTime = getTargetRuntime(nReplica, model, expKey, fast=fast)
        name = f"{model}_{expKey}_{nReplica}"
        runOutDir = suiteOutDir / name

        print("\nStarting test: ", name)
        for sweepScale in sweepScales:
            print(f"Running Submission rate {sweepScale*100}%")
            runScales = [scale*sweepScale for scale in scales]
            cmd = ['./experiment.py', '-e', 'mlperf', '-n', str(nReplica),
                   f'--runTime={runTime}', '-m', model]
            cmd += ['-s', ",".join(map(str, runScales))]
            cmd += keyToOpts(expKey)

            print("Running: ", " ".join(cmd))
            sp.run(cmd)

            shutil.copytree(resultsDir / 'latest', runOutDir / f"rate{int(sweepScale*100)}",
                            ignore=shutil.ignore_patterns("*.ipc"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Full experiment suites")
    parser.add_argument('-e', '--experiment', choices=['throughput', 'lat', 'latThr', 'mlperf', 'nshot'])
    parser.add_argument('-f', '--fast', action='store_true')
    parser.add_argument('-o', '--outDir', type=pathlib.Path)
    parser.add_argument("--hetero", action='store_true', help="For lat tests, submit using heterogeneous submission rates (based on a zipf)")

    args = parser.parse_args()

    if args.outDir is None:
        suffix = datetime.datetime.now().strftime("%d%m%y-%H%M%S")
        outDir = pathlib.Path('results') / f"{args.experiment}_suite_{suffix}" / 'run0'
        outDir.mkdir(0o700, parents=True)
    else:
        runIdx = 0
        for oldRun in args.outDir.glob('run*'):
            oldIdx = int(oldRun.name[-1])
            if oldIdx >= runIdx:
                runIdx = oldIdx + 1
        outDir = args.outDir / f'run{runIdx}'
        outDir.mkdir(0o700, parents=True)

    configs = itertools.product(models, expKeys, nReplicas)
    if args.experiment == 'throughput':
        throughput(configs, outDir, fast=args.fast)
    elif args.experiment == 'lat':
        latDistribution(configs, outDir, fast=args.fast, hetero=args.hetero)
    elif args.experiment == 'latThr':
        latThr(configs, outDir, fast=args.fast, hetero=args.hetero)
    elif args.experiment == 'mlperf':
        mlperf(configs, outDir, fast=args.fast)
    elif args.experiment == 'nshot':
        nShot(configs, outDir, fast=args.fast)
    else:
        raise ValueError("Unrecognized experiment: ", args.experiment)
