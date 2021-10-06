#!/usr/bin/env python
import subprocess as sp
import pathlib
import sys
import datetime
import shutil


def mlperfMulti(nReplica, models, modes):
    if len(sys.argv) == 1:
        suffix = datetime.datetime.now().strftime("%d%m%y-%H%M%S")
        suiteOutDir = pathlib.Path('results') / f"mlperfSuite_{suffix}"
    else:
        suiteOutDir = pathlib.Path(sys.argv[1])

    suiteOutDir.mkdir(0o700)

    resultsDir = pathlib.Path("./results")


    for model in models:
        for nReplica in nReplicas:
            for mode in modes:
                name = f"{model}_{mode}_{nReplica}"
                print("\nStarting test: ", name)
                sp.run(['./experiment.py', '-e', 'mlperfMulti', '-n', str(nReplica), '-t', mode, '-m', model])

                runOutDir = suiteOutDir / name
                shutil.copytree(resultsDir / 'latest', runOutDir, ignore=shutil.ignore_patterns("*.ipc"))


def throughput(nReplica, models, modes):
    if len(sys.argv) == 1:
        suffix = datetime.datetime.now().strftime("%d%m%y-%H%M%S")
        suiteOutDir = pathlib.Path('results') / f"throughputSuite_{suffix}"
    else:
        suiteOutDir = pathlib.Path(sys.argv[1])

    suiteOutDir.mkdir(0o700)

    resultsDir = pathlib.Path("./results")

    for model in models:
        for nReplica in nReplicas:
            for mode in modes:
                if nReplica > 4 and mode == 'tvm':
                    continue

                name = f"{model}_{mode}_{nReplica}"
                print("\nStarting test: ", name)
                sp.run(['./experiment.py',
                        '-e', 'throughput',
                        '-n', str(nReplica),
                        '-s', str(1 / nReplica),
                        '-t', mode,
                        '-m', model])

                runOutDir = suiteOutDir / name
                shutil.copytree(resultsDir / 'latest', runOutDir, ignore=shutil.ignore_patterns("*.ipc"))


# def latDistribution(nReplica, models, modes, suiteOutDir):
#     if len(sys.argv) == 1:
#         suffix = datetime.datetime.now().strftime("%d%m%y-%H%M%S")
#         suiteOutDir = pathlib.Path('results') / f"latencyDistributionSuite_{suffix}"
#     else:
#         suiteOutDir = pathlib.Path(sys.argv[1])
#
#     suiteOutDir.mkdir(0o700)
#
#     resultsDir = pathlib.Path("./results")
#
#
#     for model in models:
#         for nReplica in nReplicas:
#             for mode in modes:
#                 name = f"{model}_{mode}_{nReplica}"
#                 print("\nStarting test: ", name)
#                 sp.run(['./experiment.py', '-e', 'mlperfMulti', '-n', str(nReplica), '-t', mode, '-m', model])
#
#                 runOutDir = suiteOutDir / name
#                 shutil.copytree(resultsDir / 'latest', runOutDir, ignore=shutil.ignore_patterns("*.ipc"))

nReplicas = [1, 2, 3, 4]
models = ['resnet50', 'bert']
modes = ['kaas', 'tvm']

# mlperfMulti(nReplicas, models, modes)
throughput(nReplicas, models, modes)
