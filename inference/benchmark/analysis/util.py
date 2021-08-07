import matplotlib.pyplot as plt
import json
import sys


def plotLatencies(lats):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.hist(lats, 100, log=True)
    return fig


if __name__ == "__main__":
    resultPath = sys.argv[1]

    with open('results.json', 'r') as f:
        results = json.load(f)

    lats = results[0]['metrics']['latencies']
    fig = plotLatencies(lats)
    fig.savefig("test.png")
