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

    def throughputSingle(self, modelName, gpuType=None):
        """Return estimated single-GPU throughput of this model in ms"""
        if gpuType is None:
            gpuType = util.getGpuType()

        modelData = self.perfData['isolated'][modelName]
        return min((modelData['kaas']['qps'], modelData['tvm']['qps']))

    def latency(self, modelName, gpuType=None):
        """Return the estimated single-GPU latency of this model in ms"""
        if gpuType is None:
            gpuType = util.getGpuType()

        modelData = self.perfData['isolated'][modelName]
        return min((modelData['kaas']['latency'], modelData['tvm']['latency']))

    def throughputFull(self, modelName, nClient, modelType=None, independent=True):
        """Return throughput for the full 8 GPU experiment"""
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

    def getMlPerfConfig(self, modelName, benchConfig, gpuType=None):
        """Return an mlperf config object for the specified model"""
        if gpuType is None:
            gpuType = util.getGpuType()

        maxQps = self.throughputSingle(modelName, gpuType)
        latency = self.latency(modelName, gpuType)
        settings = model.getDefaultMlPerfCfg(maxQps, latency, benchConfig)

        return settings


def getProperties():
    global perfData
    if perfData is None:
        perfData = Properties()

    return perfData
