#!/usr/bin/env python
import os
import pathlib
import shutil
import datetime
import subprocess as sp

nIter = 10


# model = 'testModel'
model = 'resnet50'
# model = 'complexCutlassGemm'

nvCmd = ['nvprof', '-f', '--log-file', 'results.csv', '--profile-from-start', 'off', '--csv']
actNvColdCmd = nvCmd + ['./benchmark.py', '-b', 'local', '-e', 'deepProf', '--force-cold', '-m', model + "Tvm"]
actNvWarmCmd = nvCmd + ['./benchmark.py', '-b', 'local', '-e', 'deepProf', '-m', model + "Tvm"]
actPipeCmd = ['./benchmark.py', '-b', 'ray', '-e', 'nshot', '-p', 'exclusive', '-m', model + "Tvm"]
actInlineCmd = ['./benchmark.py', '-b', 'ray', '-e', 'nshot', '-p', 'exclusive', '-m', model + "Tvm", '--inline']
kaasPipeCmd = ['./benchmark.py', '-b', 'ray', '-e', 'nshot', '-p', 'balance', '-m', model + "Kaas"]


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

    print("Running nvprof Actor Cold")
    print(" ".join(actNvColdCmd))
    runTest(actNvColdCmd, nIter, suiteOutDir / "actNvCold", pathlib.Path("./results.csv"), environ)

    print("Running nvprof Actor Warm")
    print(" ".join(actNvWarmCmd))
    runTest(actNvWarmCmd, nIter, suiteOutDir / "actNvWarm", pathlib.Path("./results.csv"), environ)

    print("Running Actor Pipelined")
    print(" ".join(actPipeCmd))
    runTest(actPipeCmd, nIter, suiteOutDir / "actPipe", pathlib.Path("./results.json"), environ)

    # print("Running Actor Inlined")
    # runTest(actInlineCmd, nIter, suiteOutDir / "actInline", pathlib.Path("./results.json"), environ)

    print("Running KaaS")
    print(" ".join(kaasPipeCmd))
    runTest(kaasPipeCmd, nIter, suiteOutDir / "kaas", pathlib.Path("./results.json"), environ)


if __name__ == '__main__':
    main()
