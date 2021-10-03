#!/usr/bin/env python
import subprocess as sp
import pathlib
import sys
import datetime
import shutil
# import shutil.ignore_patterns

if len(sys.argv) == 1:
    suffix = datetime.datetime.now().strftime("%d%m%y-%H%M%S")
    suiteOutDir = pathlib.Path('results') / f"multiSuite_{suffix}"
else:
    suiteOutDir = pathlib.Path(sys.argv[1])

suiteOutDir.mkdir(0o700)

resultsDir = pathlib.Path("./results")

nReplicas = [1, 2, 3, 4]
models = ['resnet50', 'bert']
modes = ['kaas', 'tvm']

# nReplicas = [1]
# models = ['resnet50']
# modes = ['kaas']

for model in models:
    for mode in modes:
        for nReplica in nReplicas:
            # sp.run(['./experiment.py', '-e', 'mlperfMulti', '--scale=0.2', '-n', str(nReplica), '-t', mode, '-m', model])

            runOutDir = suiteOutDir / f"{model}_{mode}_{nReplica}"
            shutil.copytree(resultsDir / 'latest', runOutDir, ignore=shutil.ignore_patterns("*.ipc"))
