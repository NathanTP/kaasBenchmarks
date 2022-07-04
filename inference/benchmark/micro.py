#!/usr/bin/env python
import os
import pathlib
import shutil
import datetime
import subprocess as sp

nIter = 10


model = 'testModel'
# model = 'resnet50'

nvCmd = ['nvprof', '-f', '--log-file', 'results.csv', '--profile-from-start', 'off', '--csv']
nativeNvColdCmd = nvCmd + ['./benchmark.py', '-b', 'local', '-e', 'deepProf', '--force-cold', '-m', model, '-t', 'native']
nativeNvWarmCmd = nvCmd + ['./benchmark.py', '-b', 'local', '-e', 'deepProf', '-m', model, '-t', 'native']
nativePipeCmd = ['./experiment.py', '-b', 'ray', '-n', '1', '-e', 'nshot', '-s', '1', '-p', 'exclusive', '-m', model, '-t', "native"]
staticPipeCmd = ['./experiment.py', '-b', 'ray', '-n', '1', '-e', 'nshot', '-s', '1', '-p', 'static', '--fractional=mem', '-m', model, '-t', 'native']
actInlineCmd = ['./experiment.py', '-b', 'ray', '-n', '1', '-e', 'nshot', '-s', '1', '-p', 'exclusive', '-m', model, '-t', 'native', '--inline']
kaasPipeCmd = ['./experiment.py', '-b', 'ray', '-n', '1', '-e', 'nshot', '-s', '1', '-p', 'balance', '-m', model, '-t', "kaas"]


def runTest(cmd, niter, resDir, cmdOutPath, environ):
    if not resDir.exists():
        resDir.mkdir(mode=0o700, parents=True)

    for i in range(niter):
        proc = sp.run(cmd, env=environ, stdout=sp.PIPE, stderr=sp.STDOUT, text=True, check=True)
        shutil.copy(cmdOutPath, resDir / (f"{i}_" + cmdOutPath.name))

        with open(resDir / f"{i}.log", 'w') as f:
            f.write(proc.stdout)


def main():
    suffix = datetime.datetime.now().strftime("%d%m%y-%H%M%S")
    suiteOutDir = pathlib.Path('results') / f"micro_{suffix}"

    environ = os.environ.copy()
    environ['CUDA_VISIBLE_DEVICES'] = '0'

    print("Running nvprof Native Cold")
    print(" ".join(nativeNvColdCmd))
    runTest(nativeNvColdCmd, nIter, suiteOutDir / "nativeNvCold", pathlib.Path("./results.csv"), environ)

    print("Running nvprof Native Warm")
    print(" ".join(nativeNvWarmCmd))
    runTest(nativeNvWarmCmd, nIter, suiteOutDir / "nativeNvWarm", pathlib.Path("./results.csv"), environ)

    print("Running Native Pipelined")
    print(" ".join(nativePipeCmd))
    runTest(nativePipeCmd, nIter, suiteOutDir / "nativePipe", pathlib.Path("./results.json"), environ)

    print("Running Static Pipelined")
    print(" ".join(nativePipeCmd))
    runTest(staticPipeCmd, nIter, suiteOutDir / "staticPipe", pathlib.Path("./results.json"), environ)

    # print("Running Native Inlined")
    # runTest(nativeInlineCmd, nIter, suiteOutDir / "nativeInline", pathlib.Path("./results.json"), environ)

    print("Running KaaS")
    print(" ".join(kaasPipeCmd))
    runTest(kaasPipeCmd, nIter, suiteOutDir / "kaas", pathlib.Path("./results.json"), environ)


if __name__ == '__main__':
    main()
