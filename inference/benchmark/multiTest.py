#!/usr/bin/env python
import subprocess as sp
import pathlib
import sys
import datetime
import shutil
import time
import infbench


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
                if nReplica > 4 and mode == 'tvm':
                    continue
                time.sleep(20)
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
                # if nReplica > 6 and mode == 'tvm':
                #     continue
                time.sleep(20)

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


# These are hard coded from manually run experiments. They are the peak
# throughput achieved in the throughput suite. The slowest throughput between
# actor/tvm is used. Generate using minMaxThroughput() in analysis/util.py.
throughputBaselines = {
    'resnet50': [142.41299388906768, 141.65304793745327, 140.58463321571497,
                 141.5060638374777, 4.2258765441538975, 2.7161514543522625,
                 float("nan"), float('nan')],
    'bert': [41.29657053554684, 41.44402115116313, 31.641338978614982,
             40.58364497477458, 1.5039667823946186, 0.9752365843478145,
             float('nan'), float('nan')]
}


def getThroughputScales():
    # These are the canonical base throughputs used to determine scales
    baseThroughputs = {
        'bert': infbench.bert.bertModelBase.getPerfEstimates("Tesla V100-SXM2-16GB")[0],
        'resnet50': infbench.resnet50.resnet50Base.getPerfEstimates("Tesla V100-SXM2-16GB")[0]
    }

    # Scales should be 20% below peak throughput scale
    scaleBaselines = {name: (val / baseThroughputs[name])*0.8 for name, val in throughputBaselines.items()}
    print(scaleBaselines)

def latDistribution(nReplica, models, modes, suiteOutDir):
    if len(sys.argv) == 1:
        suffix = datetime.datetime.now().strftime("%d%m%y-%H%M%S")
        suiteOutDir = pathlib.Path('results') / f"latencyDistributionSuite_{suffix}"
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

# nReplicas = [1, 2, 3, 4, 5, 6, 7, 8]
nReplicas = [7, 8]
models = ['resnet50', 'bert']
modes = ['tvm']

throughput(nReplicas, models, modes)
# mlperfMulti(nReplicas, models, modes)
