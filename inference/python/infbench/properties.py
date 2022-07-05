import pathlib
import json
from . import model


perfData = None


class Properties():
    def __init__(self, propFile=None):
        if propFile is None:
            self.propFile = pathlib.Path(__file__).parent / 'properties.json'
        else:
            self.propFile = propFile

        with open(self.propFile, 'r') as f:
            self.perfData = json.load(f)

    def throughputSingle(self, modelName, expKey):
        """Return estimated single-GPU throughput of this model in ms"""
        # Lazy hack rather than do this right
        if expKey != 'kaas':
            expKey = 'exclusive'

        return self.perfData['isolated'][modelName][expKey]['qps']

    def resourceReqs(self, modelName, modelType):
        """Return resoure requirements for modelName and modelType (kaas or
        native). 'mem' and 'sm'.
        """
        return self.perfData['reqs'][modelName][modelType]

    def latency(self, modelName, expKey):
        """Return the estimated single-GPU latency of this model in ms"""
        # Lazy hack rather than do this right
        if expKey != 'kaas':
            expKey = 'exclusive'

        return self.perfData['isolated'][modelName][expKey]['latency']

    def throughputFull(self, modelName, nClient, expKey):
        """Return throughput for the full 8 GPU experiment"""
        return self.perfData['full'][modelName][expKey]['throughput']

    def getMlPerfConfig(self, modelName, benchConfig):
        """Return an mlperf config object for the specified model"""
        maxQps = self.throughputSingle(modelName, benchConfig['expKey'])
        latency = self.latency(modelName, benchConfig['expKey'])

        settings = model.getDefaultMlPerfCfg(maxQps, latency, benchConfig)

        return settings


def getProperties():
    global perfData
    if perfData is None:
        perfData = Properties()

    return perfData
