#!/usr/bin/env python
import subprocess as sp
import pathlib
import sys
import datetime
import shutil
import time
import infbench.bert
import infbench.resnet50


def mlperfMulti(nReplicas, models, modes):
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

    print("Final Results in: ", suiteOutDir)


def throughput(nReplicas, models, modes):
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

    print("Final Results in: ", suiteOutDir)


# These are hard coded from manually run experiments. They are the peak
# throughput achieved in the throughput suite. The slowest throughput between
# actor/tvm is used. Generate using minMaxThroughput() in analysis/util.py.
throughputBaselines = {
    'resnet50': [142.41299388906768, 141.65304793745327, 140.58463321571497,
                 141.5060638374777, 4.2258765441538975, 2.7161514543522625,
                 1.8372934445128746, 1.7213538775608817, 1.4261652856296516,
                 1.2509273341611908, 1.5632660052414507, 1.442281768453932,
                 1.3164983734181182, 1.2301623103284065, 1.1511744668537298,
                 1.083128781341447],
    'bert': [41.29657053554684, 41.44402115116313, 31.641338978614982,
             40.58364497477458, 1.5039667823946186, 0.9752365843478145,
             0.43632979442155534, 0.3499707909696742, 0.42640207944480446,
             0.3794464966505168, 0.3285686127978113, 0.3183556802632884,
             0.29756675734229465, 0.28034732132535134, 0.2516927885232863,
             0.2353052429673834]
}


def getThroughputScales():
    # These are the canonical base throughputs used to determine scales
    baseThroughputs = {
        'bert': infbench.bert.bertModelBase.getPerfEstimates("Tesla V100-SXM2-16GB")[0],
        'resnet50': infbench.resnet50.resnet50Base.getPerfEstimates("Tesla V100-SXM2-16GB")[0]
    }

    # Scales should be 20% below peak throughput scale
    scaleBaselines = {}
    for name in throughputBaselines.keys():
        scaleBaselines[name] = []
        for i, thpt in enumerate(throughputBaselines[name]):
            safeThroughput = thpt * 0.8
            totalScale = safeThroughput / baseThroughputs[name]
            scaleBaselines[name].append(totalScale / (i+1))

    return scaleBaselines


def latDistribution(nReplicas, models, modes):
    suffix = datetime.datetime.now().strftime("%d%m%y-%H%M%S")
    suiteOutDir = pathlib.Path('results') / f"latencyDistributionSuite_{suffix}"

    suiteOutDir.mkdir(0o700)

    resultsDir = pathlib.Path("./results")

    scales = getThroughputScales()
    for model in models:
        for nReplica in nReplicas:
            for mode in modes:
                name = f"{model}_{mode}_{nReplica}"
                print("\nStarting test: ", name)
                sp.run(['./experiment.py', '-e', 'mlperfMulti',
                        '-n', str(nReplica), '-s', str(scales[model][nReplica - 1]),
                        '-t', mode, '-m', model])

                runOutDir = suiteOutDir / name
                shutil.copytree(resultsDir / 'latest', runOutDir, ignore=shutil.ignore_patterns("*.ipc"))

    print("Final Results in: ", suiteOutDir)


# nReplicas = [9, 10, 11, 12, 13, 14, 15, 16]
# models = ['bert', 'resnet50']
# modes = ['kaas', 'tvm']
# throughput(nReplicas, models, modes)

# nReplicas = [1, 2, 3, 4, 5, 6, 7, 8]
nReplicas = [9, 10, 11, 12, 13, 14, 15, 16]
models = ['resnet50', 'bert']
modes = ['kaas', 'tvm']
latDistribution(nReplicas, models, modes)

# print(getThroughputScales())
# mlperfMulti(nReplicas, models, modes)
