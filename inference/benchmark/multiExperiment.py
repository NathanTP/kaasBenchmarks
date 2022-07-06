#!/usr/bin/env python
import subprocess as sp
import pathlib
import datetime
import shutil
import time
import itertools
import argparse

from infbench import properties

resultsDir = pathlib.Path("./results")

nReplicas = [1]
models = ['cGEMM', 'jacobi', 'resnet50', 'bert']
# expKeys = ['kaas', 'exclusive', 'static', 'fractional']
expKeys = ['kaas', 'exclusive']


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
            runTime = 800
        elif model == 'jacobi':
            runTime = 800
        elif model == 'resnet50':
            runTime = 600
        elif model == 'cGEMM':
            runTime = 800
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
        cmd = ['./experiment.py', '-e', 'mlperfMulti',
               '-n', str(nReplica), f'--runTime={runTime}',
               '-m', model]
        cmd += scaleArg
        cmd += keyToOpts(expKey)
        sp.run(cmd)

        runOutDir = suiteOutDir / name
        shutil.copytree(resultsDir / 'latest', runOutDir, ignore=shutil.ignore_patterns("*.ipc"))
        time.sleep(20)

    print("Final Results in: ", suiteOutDir)


def nShot(configs, suiteOutDir):
    for model, expKey, nReplica in configs:
        name = f"{model}_{expKey}_{nReplica}"
        print("\nStarting test: ", name)
        cmd = ['./experiment.py',
               '-e', 'nshot',
               '-n', str(nReplica),
               '-s', '64',
               '-m', model]
        cmd += keyToOpts(expKey)
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


def getScale(props, model, expKey, nReplica, fast):
    peakThr = props.throughputFull(model, nReplica, expKey)
    baseThr = props.throughputSingle(model, expKey)

    if fast:
        safeThr = 0.2 * peakThr
    else:
        safeThr = 0.8 * peakThr

    return (safeThr / baseThr) / nReplica


def latDistribution(configs, suiteOutDir, fast=False):
    props = properties.getProperties()
    for model, expKey, nReplica in configs:
        scale = getScale(props, model, expKey, nReplica, fast)
        runTime = getTargetRuntime(nReplica, model, expKey, fast=fast)

        name = f"{model}_{expKey}_{nReplica}"
        print("\nStarting test: ", name)
        cmd = ['./experiment.py', '-e', 'mlperf', '-n', str(nReplica),
               '-s', str(scale), f'--runTime={runTime}', '-m', model]
        cmd += keyToOpts(expKey)
        sp.run(cmd)

        runOutDir = suiteOutDir / name
        shutil.copytree(resultsDir / 'latest', runOutDir, ignore=shutil.ignore_patterns("*.ipc"))
        time.sleep(20)

    print("Final Results in: ", suiteOutDir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Full experiment suites")
    parser.add_argument('-e', '--experiment', choices=['throughput', 'lat', 'mlperf', 'nshot'])
    parser.add_argument('-f', '--fast', action='store_true')
    parser.add_argument('-o', '--outDir', type=pathlib.Path)

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
        latDistribution(configs, outDir, fast=args.fast)
    elif args.experiment == 'mlperf':
        mlperf(configs, outDir, fast=args.fast)
    elif args.experiment == 'nshot':
        nShot(configs, outDir)
    else:
        raise ValueError("Unrecognized experiment: ", args.experiment)
