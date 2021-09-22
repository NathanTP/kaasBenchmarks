import yaml
import sys
import pathlib
import pandas as pd
from pprint import pprint


def plotLatency(latData):
    resnetDf = pd.DataFrame(latData['measurements']['resnet'])
    bertDf = pd.DataFrame(latData['measurements']['bert'])


if __name__ == "__main__":
    resultPath = pathlib.Path(sys.argv[1])

    with open(resultPath / 'results.yaml', 'r') as f:
        results = yaml.safe_load(f)

    plotLatency(results['online'][0])
