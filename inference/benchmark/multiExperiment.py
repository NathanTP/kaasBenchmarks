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

# nReplicas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# models = ['bert', 'resnet50', 'jacobi', 'cGEMM']
# modes = ['kaas', 'tvm']

# nReplicas = [1, 4, 5]
# modes = ['kaas', 'tvm']

nReplicas = [1, 3, 4]
models = ['resnet50']
# models = ['bert']
modes = ['tvm']


def getTargetRuntime(nReplica, model, mode, fast=False):
    if fast:
        return 180

    # After 4 replicas, TVM is so slow that we need lots of time to get a good
    # measurement. How much longer we need depends on the model, though longer
    # is always better.
    if nReplica > 4:
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

    for model, mode, nReplica in configs:
        runTime = getTargetRuntime(nReplica, model, mode, fast=fast)

        name = f"{model}_{mode}_{nReplica}"
        print("\nStarting test: ", name)
        sp.run(['./experiment.py', '-e', 'mlperfMulti',
                '-n', str(nReplica), f'--runTime={runTime}',
                '-t', mode, '-m', model] + scaleArg)

        runOutDir = suiteOutDir / name
        shutil.copytree(resultsDir / 'latest', runOutDir, ignore=shutil.ignore_patterns("*.ipc"))
        time.sleep(20)

    print("Final Results in: ", suiteOutDir)


def nShot(configs, suiteOutDir):
    for model, mode, nReplica in configs:
        name = f"{model}_{mode}_{nReplica}"
        print("\nStarting test: ", name)
        sp.run(['./experiment.py',
                '-e', 'nshot',
                '-n', str(nReplica),
                '-s', '64',
                '-t', mode,
                '-m', model])

        runOutDir = suiteOutDir / name
        shutil.copytree(resultsDir / 'latest', runOutDir, ignore=shutil.ignore_patterns("*.ipc"))
        time.sleep(20)

    print("Final Results in: ", suiteOutDir)


def throughput(configs, suiteOutDir, fast=False):
    for model, mode, nReplica in configs:
        runTime = getTargetRuntime(nReplica, model, mode, fast=fast)

        name = f"{model}_{mode}_{nReplica}"
        print("\nStarting test: ", name)
        sp.run(['./experiment.py',
                '-e', 'throughput',
                '-n', str(nReplica),
                '-s', str(1 / nReplica),
                f'--runTime={runTime}',
                '-t', mode,
                '-m', model])

        runOutDir = suiteOutDir / name
        shutil.copytree(resultsDir / 'latest', runOutDir, ignore=shutil.ignore_patterns("*.ipc"))
        time.sleep(20)

    print("Final Results in: ", suiteOutDir)


def getScale(props, model, mode, nReplica, independent, fast):
    peakThr = props.throughputFull(model, nClient=nReplica, modelType=mode,
                                   independent=independent)
    baseThr = props.throughputSingle(model)

    if fast:
        safeThr = 0.2 * peakThr
    else:
        safeThr = 0.8 * peakThr

    return (safeThr / baseThr) / nReplica


def latDistribution(configs, suiteOutDir, independent=False, fast=False):
    props = properties.getProperties()
    for model, mode, nReplica in configs:
        scale = getScale(props, model, mode, nReplica, independent, fast)
        runTime = getTargetRuntime(nReplica, model, mode, fast=fast)

        name = f"{model}_{mode}_{nReplica}"
        print("\nStarting test: ", name)
        cmd = ['./experiment.py', '-e', 'mlperf', '-n', str(nReplica),
               '-s', str(scale), f'--runTime={runTime}', '-t', mode, '-m', model]
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
    parser.add_argument('--independent', action='store_true', help="For the lat experiment, this uses the peak throughput for each mode independently. Otherwise, the lowest throughput is used for both modes.")

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

    configs = itertools.product(models, modes, nReplicas)
    if args.experiment == 'throughput':
        throughput(configs, outDir, fast=args.fast)
    elif args.experiment == 'lat':
        latDistribution(configs, outDir, independent=args.independent, fast=args.fast)
    elif args.experiment == 'mlperf':
        mlperf(configs, outDir, fast=args.fast)
    elif args.experiment == 'nshot':
        nShot(configs, outDir)
    else:
        raise ValueError("Unrecognized experiment: ", args.experiment)
