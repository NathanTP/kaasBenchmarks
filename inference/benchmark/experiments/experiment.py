#!/usr/bin/env python
import pathlib
import subprocess as sp
import sys
import signal


expRoot = pathlib.Path(__file__).resolve().parent
benchRoot = (expRoot / "..").resolve()
resultsDir = expRoot / 'results'


def launchServer(outDir, nClient):
    """Launch the benchmark server. outDir is the directory where experiment
    outputs should go. Returns a Popen object."""
    return sp.Popen([benchRoot / "benchmark.py",
                     "-t", "server",
                     '-b', 'ray',
                     '--numClient=' + str(nClient)], cwd=outDir, stdout=sys.stdout)


def launchClient(scale, model, name, outDir):
    baseCmd = [(benchRoot / "benchmark.py"), "-t", "mlperf", "-b", "client"]
    options = ["--scale=" + str(scale), "--name=" + name, "-m", model]
    return sp.Popen(baseCmd + options, stdout=sys.stdout, cwd=outDir)


def mlperfMulti(modelType, prefix="mlperf_multi", outDir="results"):

    prefix = f"{prefix}_{modelType}"
    runners = {
        # 'resnet': launchClient(0.2, 'resnet50' + modelType, prefix+"_resnet", outDir),
        'bert': launchClient(0.2, 'bert' + modelType, prefix+"_bert", outDir),
        # 'resnet1': launchClient(0.2, 'resnet50' + modelType, prefix+"_resnet1", outDir)
        'superres': launchClient(0.2, 'superRes' + modelType, prefix+"_superRes", outDir)
    }
    server = launchServer(outDir, len(runners))

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


def mlperfOne(model, prefix="mlperfOne", outDir="results"):
    server = launchServer(outDir, 1)
    runner = launchClient(1.0, model, prefix, outDir)

    runner.wait()
    server.send_signal(signal.SIGINT)
    if runner.returncode != 0:
        print("Run Failed")
        return False
    else:
        return True


def nshotOne(model, prefix="nshotOne", outDir="results"):
    server = launchServer(outDir, 1)
    runner = launchClient(1.0, model, prefix, outDir)

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

    # if runOne('resnet50Tvm', outDir=resultsDir):
    if mlperfMulti('Kaas', outDir=resultsDir):
        print("Success")
    else:
        print("Failure")
