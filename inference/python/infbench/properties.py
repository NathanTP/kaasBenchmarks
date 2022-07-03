import pathlib
import json
from . import model
from . import util


perfData = None


class Properties():
    def __init__(self, propFile=None):
        if propFile is None:
            self.propFile = pathlib.Path(__file__).parent / 'properties.json'
        else:
            self.propFile = propFile

        with open(self.propFile, 'r') as f:
            self.perfData = json.load(f)

    def throughputSingle(self, modelName, modelType=None, gpuType=None, independent=True):
        """Return estimated single-GPU throughput of this model in ms"""
        if gpuType is None:
            gpuType = util.getGpuType()

        if modelType == 'direct':
            modelType = 'tvm'

        if independent:
            return self.perfData['isolated'][modelName][modelType]['qps']
        else:
            modelData = self.perfData['isolated'][modelName]
            return min((modelData['kaas']['qps'], modelData['tvm']['qps']))

    def resourceReqs(self, modelName, modelType):
        """Return resoure requirements for modelName and modelType (kaas or
        tvm). 'mem' and 'sm'.
        """
        if modelType == 'direct':
            modelType = 'tvm'
        return self.perfData['isolated'][modelName][modelType]

    def latency(self, modelName, modelType=None, independent=True):
        """Return the estimated single-GPU latency of this model in ms"""
        modelData = self.perfData['isolated'][modelName]
        if independent:
            return modelData[modelType]['latency']
        else:
            return min((modelData['kaas']['latency'], modelData['tvm']['latency']))

    def throughputFull(self, modelName, nClient, modelType=None, independent=True):
        """Return throughput for the full 8 GPU experiment"""
        if modelType == 'direct':
            modelType = 'tvm'

        if independent:
            thpt = self.perfData['full'][modelName][modelType]['throughput']
        else:
            kThr = self.perfData['full'][modelName]['kaas']['throughput'][nClient - 1]
            tThr = self.perfData['full'][modelName]['tvm']['throughput'][nClient - 1]
            if kThr is None or tThr is None:
                thpt = None
            else:
                thpt = min(kThr, tThr)

        if thpt is None:
            raise ValueError(f"Throughput data not available for {modelName} {nClient}")

        return thpt

    def getMlPerfConfig(self, modelName, benchConfig, gpuType=None, independent=True):
        """Return an mlperf config object for the specified model"""
        if gpuType is None:
            gpuType = util.getGpuType()

        maxQps = self.throughputSingle(modelName,
                                       modelType=benchConfig['model_type'],
                                       independent=independent)

        latency = self.latency(modelName, modelType=benchConfig['model_type'],
                               independent=independent)

        settings = model.getDefaultMlPerfCfg(maxQps, latency, benchConfig)

        return settings


def getProperties():
    global perfData
    if perfData is None:
        perfData = Properties()

    return perfData
