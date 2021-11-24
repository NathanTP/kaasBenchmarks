#!/usr/bin/env python
import os
import pathlib
import shutil
import datetime
import subprocess as sp

nIter = 2


nvCmd = ['nvprof', '-f', '--log-file', 'results.csv', '--profile-from-start', 'off', '--csv']
actNvColdCmd = nvCmd + ['./benchmark.py', '-b', 'local', '-e', 'deepProf', '--force-cold', '-m', 'testModelTvm']
actNvWarmCmd = nvCmd + ['./benchmark.py', '-b', 'local', '-e', 'deepProf', '-m', 'testModelTvm']
actPipeCmd = ['./benchmark.py', '-b', 'ray', '-e', 'nshot', '-p', 'exclusive', '-m', 'testModelTvm']
actInlineCmd = ['./benchmark.py', '-b', 'ray', '-e', 'nshot', '-p', 'exclusive', '-m', 'testModelTvm', '--inline']
kaasPipeCmd = ['./benchmark.py', '-b', 'ray', '-e', 'nshot', '-p', 'exclusive', '-m', 'testModelKaas']


def runTest(cmd, niter, resDir, cmdOutPath, environ):
    if not resDir.exists():
        resDir.mkdir(mode=0o700, parents=True)

    for i in range(niter):
        proc = sp.run(cmd, env=environ, stdout=sp.PIPE, stderr=sp.STDOUT, text=True)
        shutil.copy(cmdOutPath, resDir / (f"{i}_" + cmdOutPath.name))

        with open(resDir / f"{i}.log", 'w') as f:
            f.write(proc.stdout)


def main():
    suffix = datetime.datetime.now().strftime("%d%m%y-%H%M%S")
    suiteOutDir = pathlib.Path('results') / f"micro_{suffix}"

    environ = os.environ.copy()
    environ['CUDA_VISIBLE_DEVICES'] = '0'

    print("Running nvprof Actor Cold")
    runTest(actNvColdCmd, nIter, suiteOutDir / "actNvCold", pathlib.Path("./results.csv"), environ)

    print("Running nvprof Actor Warm")
    runTest(actNvWarmCmd, nIter, suiteOutDir / "actNvWarm", pathlib.Path("./results.csv"), environ)

    print("Running Actor Pipelined")
    runTest(actPipeCmd, nIter, suiteOutDir / "actPipe", pathlib.Path("./results.json"), environ)

    print("Running Actor Inlined")
    runTest(actInlineCmd, nIter, suiteOutDir / "actInline", pathlib.Path("./results.json"), environ)

    print("Running KaaS")
    runTest(kaasPipeCmd, nIter, suiteOutDir / "kaas", pathlib.Path("./results.json"), environ)


if __name__ == '__main__':
    main()
