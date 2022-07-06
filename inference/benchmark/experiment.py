#!/usr/bin/env python
import pathlib
import subprocess as sp
import sys
import signal
import datetime
import os
import argparse
import json
from pprint import pprint
import time

import util

expRoot = pathlib.Path(__file__).resolve().parent
resultsDir = expRoot / 'results'


def linkLatest(newLatest):
    latestDir = resultsDir / 'latest'
    latestDir.unlink(missing_ok=True)
    latestDir.symlink_to(newLatest)


class serverProc():
    def __init__(self, outDir, nClient, policy, modelType=None, nGpu=None, fractional=None):
        env = os.environ
        if nGpu is not None:
            env['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(nGpu)])

        if policy == 'static' and fractional is not None:
            self.mpsServer = sp.Popen(['nvidia-cuda-mps-control', '-f'], stdout=sp.DEVNULL)
            time.sleep(2)
        else:
            self.mpsServer = None

        cmd = [expRoot / "benchmark.py",
                         "-e", "server",
                         '-b', 'ray',
                         '-t', modelType,
                         '--policy=' + policy,
                         '--numClient=' + str(nClient)]

        if fractional is not None:
            fracArgs = ['--fractional', fractional]
            cmd += fracArgs

        self.benchServer = sp.Popen(cmd, cwd=outDir, stdout=sys.stdout, env=env)

    def stop(self):
        if self.mpsServer is not None:
            self.mpsServer.send_signal(signal.SIGINT)
            self.mpsServer.wait()

        self.benchServer.send_signal(signal.SIGINT)
        self.benchServer.wait()


# There are a lot of parameters to the clients that don't make a functional
# difference since only the server uses them (e.g. fractional, policy, etc.).
# However, we include them so that the client logs can include the full
# configuration information.
def launchClient(scale, model, modelType, name, test, outDir, runTime=None,
                 nClient=1, fractional=None, policy=None):
    cmd = [(expRoot / "benchmark.py"),
           "-e", test,
           "-b", "client",
           "--numClient", str(nClient),
           "--name=" + name,
           "-m", model,
           "-p", policy,
           "--modelType", modelType]

    if scale is not None:
        cmd.append("--scale=" + str(scale))

    if runTime is not None:
        cmd.append("--runTime=" + str(runTime))

    if fractional is not None:
        fracArgs = ['--fractional', fractional]
        cmd += fracArgs

    return sp.Popen(cmd, stdout=sys.stdout, cwd=outDir)


def runTest(test, modelNames, modelType, prefix, resultsDir, nCpy=1, scales=None,
            runTime=None, policy=None, fractional=None):
    """Run a single test in client-server mode.
        modelNames: Models to run. At least one copy of these models will run
        nCpy: Number of copies of modelNames to run. len(modelNames)*nCpy clients will run
        modelType: kaas or native
        prefix: Used to name the run
        resultsDir: Where to output all results from this test
        scale: For tests that use the --scale parameter
        runTime: Target runtime of experiment
        policy: Scheduling policy to use for this experiment
        fractional: See argparse help for details.
    """
    if scales is None:
        scales = [1]*nCpy*len(modelNames)

    runners = {}
    for i in range(nCpy):
        for j, modelName in enumerate(modelNames):
            instanceName = f"{prefix}_{modelName}_{j}_{i}"
            runners[instanceName] = launchClient(scales[j], modelName,
                                                 modelType, instanceName, test,
                                                 resultsDir, runTime=runTime,
                                                 nClient=nCpy*len(modelNames),
                                                 fractional=fractional, policy=policy)

    server = serverProc(resultsDir, len(runners), policy,
                        fractional=fractional, modelType=modelType)

    failed = []
    for name, runner in runners.items():
        runner.wait()
        if runner.returncode != 0:
            failed.append(name)
    server.stop()

    if len(failed) != 0:
        raise RuntimeError("Some runners failed: ", failed)

    if test == 'mlperf':
        for summaryFile in resultsDir.glob("*_summary.txt"):
            with open(summaryFile, 'r') as f:
                summary = f.read()
            if "INVALID" in summary:
                if "Min queries satisfied : NO" in summary:
                    # raise RuntimeError("Test didn't meet minimum queries, try again with a longer runtime")
                    print("WARNING: Test didn't meet minimum queries, try again with a longer runtime")
                if "Min duration satisfied : NO" in summary:
                    # raise RuntimeError("Test didn't meet minimum duration, try again with a longer runtime")
                    print("WARNING: Test didn't meet minimum duration, try again with a longer runtime")

                return False

    return True


def mlperf(modelType, outDir="results", scales=None, runTime=None, nCpy=1,
           models=None, policy=None, expKey=None, fractional=None):
    suffix = datetime.datetime.now().strftime("%d%m%y-%H%M%S")

    prefix = "mlperf_" + expKey

    expResultsDir = outDir / f"{prefix}_{suffix}"
    expResultsDir.mkdir(0o700)
    linkLatest(expResultsDir)

    if scales is None:
        raise ValueError("mlperf requires explicit scales")

    runTest('mlperf', models, modelType, prefix, expResultsDir,
            scales=scales, runTime=runTime, nCpy=nCpy, policy=policy,
            fractional=fractional)

    p50s = {}
    p90s = {}
    for resultsFile in expResultsDir.glob("**/mlperf*_results.json"):
        with open(resultsFile, 'r') as f:
            result = json.load(f)

        p50s[result['config']['name']] = result['metrics_warm']['t_response']['p50']
        p90s[result['config']['name']] = result['metrics_warm']['t_response']['p90']

    print("p50s:")
    pprint(p50s)
    print("p90s:")
    pprint(p90s)


# WARNING: This isn't really used for anything in the current experiments and
# is not maintained. Don't expect it to work the first time if you need it
# again.
# def mlperfSearch(modelType, prefix="mlperf_multi", outDir="results", scale=None,
#                  runTime=None, nCpy=1, models=None, policy=None):
#     suffix = datetime.datetime.now().strftime("%d%m%y-%H%M%S")
#     expResultsDir = outDir / f"mlperf_{modelType}_{suffix}"
#
#     prefix = f"{prefix}_{modelType}"
#
#     # Attempt to find a valid scale, starting with "perfect" scaling
#     nModel = nCpy * len(models)
#     if scale is None:
#         scale = ((1 / nModel) * infbench.getNGpu())
#         startScale = scale
#         succeedScale = 0
#         failureScale = scale
#         runOnce = False
#     else:
#         # This tricks the system into only running one iteration
#         startScale = float('inf')
#         succeedScale = scale
#         failureScale = scale
#         runOnce = True
#
#     # Minimum step size when searching
#     step = 0.05
#
#     # Binary Search
#     found = False
#     while not found:
#         print("\n\nAttempting scale: ", scale)
#         time.sleep(10)  # ray is bad at cleaning up, gotta wait to be sure
#         with tempfile.TemporaryDirectory() as tmpRes:
#             tmpRes = pathlib.Path(tmpRes)
#             failure = not runTest('mlperf', models, modelType, prefix, tmpRes,
#                                   nCpy=nCpy, scale=scale, runTime=runTime,
#                                   policy=policy)
#             if failure:
#                 failureScale = scale
#             else:
#                 if expResultsDir.exists():
#                     shutil.rmtree(expResultsDir)
#                 shutil.copytree(tmpRes, expResultsDir, ignore=shutil.ignore_patterns("*.ipc"))
#                 succeedScale = scale
#
#             if (failureScale - succeedScale) <= step:
#                 # Sometimes we guess wrong and start too low, this bumps us up a
#                 # bit to make sure we get a valid answer.
#                 if scale == startScale and not runOnce:
#                     scale *= 1.5
#                     startScale = scale
#                 else:
#                     found = True
#                     # If we never found a passing result, we just report the last
#                     # one that ran
#                     if not expResultsDir.exists():
#                         shutil.copytree(tmpRes, expResultsDir, ignore=shutil.ignore_patterns("*.ipc"))
#
#             else:
#                 scale = succeedScale + ((failureScale - succeedScale) / 2)
#
#     linkLatest(expResultsDir)
#
#     print("Final results at: ", expResultsDir)
#     print("Max achievable scale: ", scale)
#     return succeedScale


def nShot(n, modelType='kaas', nCpy=1, outDir="results", models=None,
          policy=None, fractional=None, expKey=None):
    suffix = datetime.datetime.now().strftime("%d%m%y-%H%M%S")
    expResultsDir = outDir / f"nshot_{expKey}_{suffix}"
    expResultsDir.mkdir(0o700)
    linkLatest(expResultsDir)

    prefix = "nshot_" + expKey

    runTest('nshot', models, modelType, prefix, expResultsDir, scales=[n]*len(models),
            nCpy=nCpy, policy=policy, fractional=fractional)


def throughput(modelType, runTime=None, outDir="results",
               nCpy=1, models=None, policy=None, fractional=None, expKey=None):
    suffix = datetime.datetime.now().strftime("%d%m%y-%H%M%S")

    prefix = "throughput_" + expKey

    expResultsDir = outDir / f"{prefix}_{suffix}"
    expResultsDir.mkdir(0o700)
    linkLatest(expResultsDir)

    runTest('throughput', models, modelType, prefix, expResultsDir,
            runTime=runTime, nCpy=nCpy, policy=policy, fractional=fractional)

    results = {}
    for resultsFile in expResultsDir.glob("throughput_*_results.json"):
        with open(resultsFile, 'r') as f:
            result = json.load(f)

        results[result['config']['name']] = result['metrics_warm']['throughput']['mean']

    print("Aggregated Results:")
    pprint(results)


if __name__ == "__main__":
    if not resultsDir.exists():
        resultsDir.mkdir(0o700)

    parser = argparse.ArgumentParser("Experiments for the benchmark")
    parser.add_argument("-m", "--model", action='append',
                        choices=['testModel', 'bert', 'resnet50', 'superRes', 'cGEMM', 'jacobi'],
                        help="Model to run. Not used in mlperfMulti mode.")
    parser.add_argument("-e", "--experiment",
                        choices=['nshot', 'mlperf', 'throughput'],
                        help="Which experiment to run.")
    parser.add_argument("-t", "--modelType", default='native',
                        choices=['kaas', 'native'], help="Which model type to use")
    parser.add_argument("-s", "--scale", type=float, action='append', help="For mlperf modes, what scale to run each client at. If omitted, tests will try to find peak performance. For nshot, this is the number of iterations. If there are multiple models, multiple scales can be provided for each model (just specify the model and scale in the same order).")
    parser.add_argument("--runTime", type=float, help="Target runtime for experiment in seconds (only valid for throughput and mlperf tests).")
    parser.add_argument("-n", "--nCopy", type=int, default=1, help="Number of model replicas to use")
    parser.add_argument("-p", "--policy", choices=['exclusive', 'balance', 'static', 'affinity'],
                        help="Scheduling policy to use. If omitted, the default policy for the model type will be used (Exclusive for native, Balance for kaas)")
    parser.add_argument("--fractional", default=None, choices=['mem', 'sm'],
                        help="In server mode, assign fractional GPUs to clients based on the specified resource (memory or SM)")

    args = parser.parse_args()

    if args.scale is None:
        args.scale = [1]*len(args.model)
    elif len(args.scale) == 1:
        args.scale = args.scale*len(args.model)

    if args.policy is None:
        if args.modelType == 'kaas':
            args.policy = 'affinity'
        elif args.modelType == 'native':
            args.policy = 'exclusive'
        else:
            raise ValueError("Unrecognized model type: ", args.modelType)

    benchConfig = util.argsToConfig(args)

    if args.experiment == 'nshot':
        print("Starting nshot")
        nShot(int(args.scale[0]), modelType=args.modelType, nCpy=args.nCopy,
              outDir=resultsDir, models=args.model, policy=args.policy,
              fractional=args.fractional, expKey=benchConfig['expKey'])
    elif args.experiment == 'mlperf':
        print("Starting mlperf")
        mlperf(args.modelType, outDir=resultsDir, scales=args.scale,
               runTime=args.runTime, models=args.model, nCpy=args.nCopy,
               policy=args.policy, expKey=benchConfig['expKey'],
               fractional=args.fractional)
    elif args.experiment == 'throughput':
        print("Starting Throughput Test")
        throughput(args.modelType, outDir=resultsDir, runTime=args.runTime,
                   models=args.model, nCpy=args.nCopy, policy=args.policy,
                   fractional=args.fractional, expKey=benchConfig['expKey'])
    else:
        raise ValueError("Invalid experiment: ", args.experiment)

    print("DONE")
