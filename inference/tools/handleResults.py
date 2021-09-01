#!/usr/bin/env python3
import sys
import json
from pprint import pprint


def kaasBreakdown(stats):
    # pprint({k: v['mean'] for k, v in stats['metrics'].items()})
    # pprint(list(stats['metrics'].keys()))

    stats = {k: v['mean'] for k, v in stats.items()}

    report = {}
    report['Read Inputs'] = stats['worker:t_hostDLoad']
    report['CUDA MM'] = stats['worker:t_cudaMM']
    report['Zero Buffers'] = stats['worker:t_zero']
    report['Load Inputs'] = stats['worker:t_htod']
    report['Kernel'] = stats['worker:t_invoke']
    report['Copy Results'] = stats['worker:t_dtoh']
    report['Write Results'] = stats['worker:t_hostDWriteBack']
    report['Other'] = stats['worker:t_e2e'] - sum(report.values())

    return report


def main():
    with open(sys.argv[1], 'r') as f:
        results = json.load(f)

    pprint(kaasBreakdown(results[0]['metrics']))


main()
