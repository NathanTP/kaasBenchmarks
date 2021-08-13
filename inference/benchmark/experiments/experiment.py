#!/usr/bin/env python
import pathlib
import subprocess as sp
import sys
import signal


expRoot = pathlib.Path(__file__).resolve().parent
benchRoot = (expRoot / "..").resolve()
resultsDir = expRoot / 'results'


def launchServer(outDir, nClient, modelType, policy):
    """Launch the benchmark server. outDir is the directory where experiment
    outputs should go. Returns a Popen object."""
    if modelType == 'Kaas':
        modeArg = '--runner_mode=kaas'
    elif modelType == 'Tvm':
        modeArg = '--runner_mode=actor'
    else:
        raise ValueError("Unrecognized model type: " + modelType)

    return sp.Popen([benchRoot / "benchmark.py",
                     "-t", "server",
                     '-b', 'ray',
                     modeArg,
                     '--runner_policy=' + policy,
                     '--numClient=' + str(nClient)], cwd=outDir, stdout=sys.stdout)


def launchClient(scale, model, name, test, outDir, nIter=1):
    cmd = [(benchRoot / "benchmark.py"),
           "-t", test,
           "--numRun=" + str(nIter),
           "-b", "client",
           "--name=" + name,
           "-m", model]

    if scale is not None:
        cmd.append("--scale=" + str(scale))

    return sp.Popen(cmd, stdout=sys.stdout, cwd=outDir)


def mlperfMulti(modelType, prefix="mlperf_multi", outDir="results"):

    if modelType == 'Kaas':
        policy = 'balance'
    elif modelType == 'Tvm':
        policy = 'exclusive'
    else:
        raise ValueError("Unrecognized Model Type: " + modelType)

    prefix = f"{prefix}_{modelType}"

    # scale = 0.05
    scale = 0.1
    runners = {
        'resnet': launchClient(scale, 'resnet50' + modelType, prefix+"_resnet", 'mlperf', outDir),
        'bert': launchClient(scale, 'bert' + modelType, prefix+"_bert", 'mlperf', outDir),
        # 'bert1': launchClient(scale, 'bert' + modelType, prefix+"_bert1", 'mlperf', outDir),
        'bert2': launchClient(scale, 'bert' + modelType, prefix+"_bert2", 'mlperf', outDir)
        # 'resnet1': launchClient(scale, 'resnet50' + modelType, prefix+"_resnet1", 'mlperf', outDir)
        # 'superres': launchClient(scale, 'superRes' + modelType, prefix+"_superRes", 'mlperf', outDir)
    }
    server = launchServer(outDir, len(runners), modelType, policy)

    failed = []
    for name, runner in runners.items():
        runner.wait()
        if runner.returncode != 0:
            failed.append(name)
    server.send_signal(signal.SIGINT)

    if len(failed) != 0:
        print("Some runners failed: ", failed)
        return False
    else:
        return True


def mlperfOne(baseModel, modelType, prefix="mlperfOne", outDir="results"):
    server = launchServer(outDir, 1, modelType, 'balance')

    model = baseModel + modelType
    # runner = launchClient(None, model, prefix, 'mlperf', outDir)
    runner = launchClient(1.0, model, prefix, 'mlperf', outDir)

    runner.wait()
    server.send_signal(signal.SIGINT)
    if runner.returncode != 0:
        print("Run Failed")
        return False
    else:
        return True


def nShot(baseModel, modelType, nIter=1, prefix="nshotOne", outDir="results"):
    server = launchServer(outDir, 1, modelType, 'balance')

    model = baseModel + modelType
    runner = launchClient(1.0, model, prefix, 'nshot', outDir, nIter=nIter)

    runner.wait()
    server.send_signal(signal.SIGINT)
    if runner.returncode != 0:
        print("Run Failed")
        return False
    else:
        return True


if __name__ == "__main__":
    if not resultsDir.exists():
        resultsDir.mkdir(0o700)

    # if nShot('superres', 'Tvm', outDir=resultsDir, nIter=32):
    # if mlperfOne('resnet50', 'Tvm', outDir=resultsDir):
    if mlperfMulti('Kaas', outDir=resultsDir):
        print("Success")
    else:
        print("Failure")
