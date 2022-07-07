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

    def latency(self, modelName, expKey, e2e=True):
        """Return the estimated single-GPU latency of this model in ms.
        Arguments
            e2e: Return the end-to-end runtime from the clients perspective. If
            false, returns the model runtime as seen by the pool.
        """
        # Lazy hack rather than do this right
        if expKey != 'kaas':
            expKey = 'exclusive'

        if e2e:
            return self.perfData['isolated'][modelName][expKey]['latency']
        else:
            return self.perfData['isolated'][modelName][expKey]['model_runtime']

    def throughputFull(self, modelName, nClient, expKey, independent=True):
        """Return throughput for the full 8 GPU experiment"""
        if not independent and (expKey == 'kaas' or expKey == 'exclusive'):
            kThpt = self.perfData['full'][modelName]['kaas']['throughput'][nClient - 1]
            fThpt = self.perfData['full'][modelName]['exclusive']['throughput'][nClient - 1]
            return min(kThpt, fThpt)
        else:
            return self.perfData['full'][modelName][expKey]['throughput'][nClient - 1]

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
