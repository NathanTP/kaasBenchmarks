#!/usr/bin/env python
import subprocess as sp
import pathlib
import sys
import datetime
import shutil
import time
import numpy as np
import itertools
import infbench.bert
import infbench.resnet50
import infbench.jacobi
import infbench.complexCutlassGemm


def getTargetRuntime(nReplica, model, mode):
    # After 4 replicas, TVM is so slow that we need lots of time to get a good
    # measurement. How much longer we need depends on the model, though longer
    # is always better.
    if nReplica > 4:
        if model == 'bert':
            runTime = 1500
        elif model == 'jacobi':
            runTime = 800
        elif model == 'resnet50':
            runTime = 600
        elif model == 'complexCutlassGemm':
            runTime = 800
        else:
            raise RuntimeError("Please configure a target runtime for model: ", model)
    else:
        runTime = 300

    return runTime


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
                runTime = getTargetRuntime(nReplica, model, mode)

                time.sleep(20)
                name = f"{model}_{mode}_{nReplica}"
                print("\nStarting test: ", name)
                sp.run(['./experiment.py', '-e', 'mlperfMulti',
                        '-n', str(nReplica), f'--runTime={runTime}',
                        '-t', mode, '-m', model])

                runOutDir = suiteOutDir / name
                shutil.copytree(resultsDir / 'latest', runOutDir, ignore=shutil.ignore_patterns("*.ipc"))

    print("Final Results in: ", suiteOutDir)


def throughput(configs):
    if len(sys.argv) == 1:
        suffix = datetime.datetime.now().strftime("%d%m%y-%H%M%S")
        suiteOutDir = pathlib.Path('results') / f"throughputSuite_{suffix}"
    else:
        suiteOutDir = pathlib.Path(sys.argv[1])

    suiteOutDir.mkdir(0o700)

    resultsDir = pathlib.Path("./results")

    for model, mode, nReplica in configs:
        time.sleep(20)
        runTime = getTargetRuntime(nReplica, model, mode)

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

    print("Final Results in: ", suiteOutDir)


# These are hard coded from manually run experiments. They are the peak
# throughput achieved in the throughput suite. The slowest throughput between
# actor/tvm is used. Generate using getMaxThroughputs() in analysis/util.py.
throughputBaselines = {
    'resnet50': {
        'tvm': [167.94983615487527, 167.890368559212, 166.83422896918697,
                167.77963429296392, 4.2258765441538975, 2.7161514543522625,
                1.8372934445128746, 1.7213538775608817, 1.4261652856296516,
                1.2509273341611908, 1.5632660052414507, 1.442281768453932,
                1.3164983734181182, 1.2301623103284065, 1.1511744668537298,
                1.083128781341447],
        'kaas': [142.41299388906768, 141.65304793745327, 140.58463321571497,
                 141.5060638374777, 140.79477360408, 140.17887091856926,
                 140.58491848851642, 140.46099051498413, 140.7460789242818,
                 138.70948093261256, 140.39874997731908, 139.19418740433724,
                 138.92141368202894, 139.06579383344578, 138.99067643572522,
                 138.02049818832566]},
    'bert': {
        'tvm': [41.29657053554684, 42.10513483779364, 31.641338978614982,
                42.03383422252679, 1.5039667823946186, 0.9752365843478145,
                0.43632979442155534, 0.3499707909696742, 0.42640207944480446,
                0.3794464966505168, 0.3285686127978113, 0.3183556802632884,
                0.29756675734229465, 0.28034732132535134, 0.2516927885232863,
                0.2353052429673834],
        'kaas': [41.658510346909885, 41.44402115116313, 41.1302849699865,
                 40.58364497477458, 40.48919750058649, 39.692854930990826,
                 39.57202635239513, 39.58878468418513, 39.561526322955615,
                 38.63981988493197, 26.49509544161356, 21.171283835280583,
                 16.88707142480664, 14.607446681579834, 13.480224461065045,
                 12.437228484158275]},
    'jacobi': {
        'tvm': [66.91134999636802, 68.16615391047895, 51.0122899139404,
                67.70568871999389, 3.6463564419018937, 1.8912824361328877,
                1.6358352122688213, 1.1126111488326782, 1.2356571121036493,
                1.1811356458061184, 1.0364348558750922, 0.9355437883246753,
                0.924301661755399, 0.9033813320002326, 0.7387523549948113,
                0.7885506437489563],
        'kaas': [45.05692180522658, 44.77894955631741, 45.297708547226236,
                 44.591042638752334, 44.19646422483872, 43.083259320077424,
                 44.38489515651817, 43.98021311874452, 43.51220290525323,
                 42.50994612503706, 41.778837485314455, 41.17097591632982,
                 41.39473714769959, 40.359503014310285, 41.26309739381903,
                 39.72380394001538]},
    'complexCutlassGemm': {
        'tvm': [90.85681792081911, 92.81105124310236, 69.59237613396404,
                91.92402379253754, 1.8159209527599, 1.263513424824398,
                1.0391331406712292, 0.9584419107765858, 0.9188624188604786,
                0.8155118579759689, 0.9426866422180428, 0.8600994224519808,
                0.7922329906486072, 0.7359534006386524, 0.6785265126805546,
                0.6613894658046942],
        'kaas': [89.49723638458205, 89.50440521157952, 89.1882472364158,
                 89.03848861133442, 89.35887427866976, 89.44364918180077,
                 89.50125584906124, 49.83026619858238, 28.673893737404637,
                 22.435888649366834, 19.210344056836572, 16.759260902476257,
                 15.21827104284447, 13.848851591942084, 13.348677117763334,
                 12.181386330955116]}
}


def getMinMaxThroughputs():
    minMax = {}
    for model, throughputs in throughputBaselines.items():
        minMax[model] = np.minimum(throughputs['tvm'], throughputs['kaas'])

    return minMax


def getThroughputScales(independent=False):
    # These are the canonical base throughputs used to determine scales
    baseThroughputs = {
        'bert': infbench.bert.bertModelBase.getPerfEstimates("Tesla V100-SXM2-16GB")[0],
        'resnet50': infbench.resnet50.resnet50Base.getPerfEstimates("Tesla V100-SXM2-16GB")[0],
        'jacobi': infbench.jacobi.jacobiBase.getPerfEstimates("Tesla V100-SXM2-16GB")[0],
        'complexCutlassGemm': infbench.complexCutlassGemm.sgemmBase.getPerfEstimates("Tesla V100-SXM2-16GB")[0]
    }

    if independent:
        throughputs = throughputBaselines
    else:
        throughputs = {}
        minMaxs = getMinMaxThroughputs()
        for model in throughputBaselines.keys():
            throughputs[model] = {}
            throughputs[model]['kaas'] = minMaxs[model]
            throughputs[model]['tvm'] = minMaxs[model]

    scales = {}
    for model, modelThpt in throughputs.items():
        scales[model] = {}
        for mode, modeThpt in modelThpt.items():
            scales[model][mode] = []
            for i, thpt in enumerate(modeThpt):
                safeThroughput = thpt * 0.8
                totalScale = safeThroughput / baseThroughputs[model]
                scales[model][mode].append(totalScale / (i+1))

    return scales


def latDistribution(configs, independent=False):
    suffix = datetime.datetime.now().strftime("%d%m%y-%H%M%S")
    if independent:
        suiteOutDir = pathlib.Path('results') / f"latencyIndependentSuite_{suffix}"
    else:
        suiteOutDir = pathlib.Path('results') / f"latencyDistributionSuite_{suffix}"

    suiteOutDir.mkdir(0o700)

    resultsDir = pathlib.Path("./results")

    scales = getThroughputScales(independent=independent)
    for model, mode, nReplica in configs:
        scale = scales[model][mode][nReplica - 1]
        runTime = getTargetRuntime(nReplica, model, mode)

        name = f"{model}_{mode}_{nReplica}"
        print("\nStarting test: ", name)
        sp.run(['./experiment.py', '-e', 'mlperfMulti',
                '-n', str(nReplica), '-s', str(scale), f'--runTime={runTime}',
                '-t', mode, '-m', model])

        runOutDir = suiteOutDir / name
        shutil.copytree(resultsDir / 'latest', runOutDir, ignore=shutil.ignore_patterns("*.ipc"))

    print("Final Results in: ", suiteOutDir)


# nReplicas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# models = ['bert', 'resnet50', 'jacobi']
# modes = ['kaas', 'tvm']

# nReplicas = [1, 4, 5, 16]
# nReplicas = [1, 2, 5]
# nReplicas = [1, 3, 5, 16]
nReplicas = [4]
# nReplicas = [2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# models = ['bert', 'jacobi']
models = ['complexCutlassGemm']
modes = ['tvm', 'kaas']
configs = itertools.product(models, modes, nReplicas)

latDistribution(configs, independent=False)
# throughput(configs)
