import numpy as np
import json
import re
import subprocess as sp
import os
import pathlib
from pprint import pprint
from kaas import profiling


cudaRT = None
profilerStarted = False


def getGpuType():
    """Return a string describing the first available GPU"""
    proc = sp.run(['nvidia-smi', '-L'], text=True, stdout=sp.PIPE, check=True)
    match = re.search(r".*: (.*) \(UUID", proc.stdout)
    return match.group(1)


nGpu = None


def getNGpu():
    """Returns the number of available GPUs on this machine"""
    global nGpu
    if nGpu is None:
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            nGpu = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        else:
            proc = sp.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                          stdout=sp.PIPE, text=True, check=True)
            nGpu = proc.stdout.count('\n')

    return nGpu


def parseMlPerf(prefix):
    with open(prefix + "summary.txt", 'r') as f:
        mlLog = f.readlines()

    metrics = profiling.profCollection()

    scheduledPattern = re.compile("Scheduled samples per second : (.*)$")
    completedPattern = re.compile("Completed samples per second    : (.*)$")
    validPattern = re.compile(".*INVALID$")

    # metrics['valid'] = True
    valid = True
    for idx, line in enumerate(mlLog):
        match = scheduledPattern.match(line)
        if match is not None:
            metrics['submission_rate'].increment(float(match.group(1)))
            # metrics['submission_rate'] = float(match.group(1))
            continue

        match = completedPattern.match(line)
        if match is not None:
            metrics['completion_rate'].increment(float(match.group(1)))
            # metrics['completion_rate'] = float(match.group(1))
            continue

        match = validPattern.match(line)
        if match is not None:
            valid = False
            continue

        if line == "Test Parameters Used\n":
            break

    if 'submission_rate' not in metrics or 'completion_rate' not in metrics:
        raise RuntimeError("Failed to parse mlperf log: ", prefix + "summary.txt")

    return metrics, valid


def processLatencies(benchConfig, rawLatencies, outPath="./results.json", mlPerfPrefix="mlperf_log_"):
    """Reads latencies from mlperf and generates both human and machine
    readable reports."""

    # latencies is a list of latencies for each query issued (in ns).
    lats = np.array(rawLatencies, dtype=np.float32)

    lats = np.divide(lats, 1E9)

    metrics = profiling.prof(fromDict={'events': lats.tolist(), 'total': float(lats.sum()), 'nevent': len(lats)})

    return metrics


def saveReport(warmMetrics, coldMetrics, benchConfig, outPath):
    if not isinstance(outPath, pathlib.Path):
        outPath = pathlib.Path(outPath).resolve()

    if warmMetrics is None and coldMetrics is None:
        raise ValueError("warmMetrics and coldMetrics cannot both be None")

    if outPath.exists():
        outPath.unlink()

    record = {
        "config": benchConfig,
        "metrics_warm": warmMetrics,
        "metrics_cold": coldMetrics
    }

    print("Saving metrics to: ", outPath)
    with open(outPath, 'w') as f:
        json.dump(record, f)


def printReport(report, metrics=None):
    cleanReport = {}
    for statName, statValues in report.items():
        if metrics is None:
            metrics = statValues.keys()

        cleanReport[statName] = {metric: value for metric, value in statValues.items() if metric in metrics}

    pprint(cleanReport)
