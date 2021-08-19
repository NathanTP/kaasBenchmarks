import infbench
import util

import mlperf_loadgen
import threading
import queue
from gpuinfo import GPUInfo
from pprint import pprint
import json
import pathlib

import libff as ff
import libff.invoke
import libff.kv
import libff.kaas as kaas
import libff.kaas.kaasFF

# for cuda profiling
import pycuda.driver as cuda


def _getHandlers(modelSpec):
    loader = modelSpec.loader(modelSpec.dataDir)

    # Create as many models as we have GPUs to get some concurrency. The local
    # mode doesn't independently scale pre/post/run.
    models = []
    for i in range(len(GPUInfo.check_empty())):
        models.append(modelSpec.modelClass(modelSpec.getModelArg()))

    return (loader, models)


# KaaS requires an immutable object store. This means that keys must be unique.
# This counter ensures that
kaasNextId = 0


def runKaas(model, kaasCtx, kaasHandle, constants, inputs, preOut):
    """Run a kaas model. inputs and preOut are literal values, constants should
    be keys in kaasCtx.kv for the constants"""
    global kaasNextId

    inputKeys = []
    for inp in inputs:
        kaasCtx.kv.put(kaasNextId, inp)
        inputKeys.append(kaasNextId)
        kaasNextId += 1

    preKeys = []
    for val in preOut:
        kaasCtx.kv.put(kaasNextId, val)
        preKeys.append(kaasNextId)
        kaasNextId += 1

    outKeys = []
    for output in model.meta['outputs']:
        outKeys.append((output['name'], kaasNextId))
        kaasNextId += 1

    runInp = util.packInputs(model.runMap, const=constants, inp=inputKeys, pre=preKeys)
    req = model.run(runInp, outKeys=outKeys)

    kaasHandle.Invoke(req.toDict())

    outputs = []
    for name, key in outKeys:
        outputs.append(kaasCtx.kv.get(key))

    return outputs


def runNative(model, constants, inputs, preOut):
    runInp = util.packInputs(model.runMap, const=constants, inp=inputs, pre=preOut)
    return model.run(runInp)


def _runOne(model, constants, inputs, stats=None, kaasCtx=None, kaasHandle=None, constKeys=None):
    with util.timer("pre", stats):
        preInp = util.packInputs(model.preMap, const=constants, inp=inputs)
        preOut = model.pre(preInp)

    with util.timer("run", stats):
        if kaasCtx is not None:
            runOut = runKaas(model, kaasCtx, kaasHandle, constKeys, inputs, preOut)
        else:
            runOut = runNative(model, constants, inputs, preOut)

    if model.noPost:
        postOut = runOut
    else:
        with util.timer("post", stats):
            postInp = util.packInputs(model.postMap, const=constants, inp=inputs, pre=preOut, run=runOut)
            postOut = model.post(postInp)

    return postOut


def nShot(modelSpec, n, benchConfig, reportPath="results.json"):
    loader, models = _getHandlers(modelSpec)
    stats = util.profCollection()

    loader.preLoad(list(range(min(n, loader.ndata))))
    model = models[0]
    constants = model.getConstants(modelSpec.modelPath.parent)

    if modelSpec.modelType == "kaas":
        objStore = ff.kv.Local(copyObjs=False, serialize=False)
        kaasCtx = ff.invoke.RemoteCtx(None, objStore)
        kaasHandle = kaas.kaasFF.getHandle('direct', kaasCtx, stats=stats)

        global kaasNextId
        constKeys = []
        for const in constants:
            kaasCtx.kv.put(kaasNextId, const)
            constKeys.append(kaasNextId)
            kaasNextId += 1
    else:
        kaasCtx = None
        kaasHandle = None
        constKeys = None

    # Cold starts
    # for i in range(util.getNGpu()):
    for i in range(1):
        inputs = loader.get(0)
        _runOne(model, constants, inputs, stats=stats, kaasCtx=kaasCtx, kaasHandle=kaasHandle, constKeys=constKeys)

    accuracies = []
    results = []
    for i in range(n):
        idx = i % loader.ndata
        inputs = loader.get(idx)

        cuda.start_profiler()
        with util.timer("t_e2e", stats):
            result = _runOne(model, constants, inputs, stats=stats, kaasCtx=kaasCtx, kaasHandle=kaasHandle, constKeys=constKeys)
        cuda.stop_profiler()

        results.append(result)

        if loader.checkAvailable:
            accuracies.append(loader.check(result, idx))

    if loader.checkAvailable:
        print("Accuracy = ", sum([int(res) for res in accuracies]) / n)
    else:
        print("Dataset does not support accuracy calculation")

    report = stats.report()
    print("E2E Results:")
    pprint({(k, v) for (k, v) in report['t_e2e'].items() if k != "events"})

    if not isinstance(reportPath, pathlib.Path):
        reportPath = pathlib.Path(reportPath).resolve()

    print("Saving results to: ", reportPath)
    if reportPath.exists():
        with open(reportPath, 'r') as f:
            fullReport = json.load(f)
    else:
        fullReport = []

    record = {
        "config": benchConfig,
        "metrics": report
    }
    fullReport.append(record)

    with open(reportPath, 'w') as f:
        json.dump(fullReport, f)

    return results


# =============================================================================
# MLPERF INFERENCE STUFF
# =============================================================================

class mlperfRunner():

    def __init__(self, loader, constants, models, benchConfig):
        self.loader = loader
        self.models = models
        self.queue = queue.SimpleQueue()
        self.constants = constants
        self.benchConfig = benchConfig

    def start(self):
        for model in self.models:
            threading.Thread(target=self.runOneAsync, args=[model, self.queue]).start()

    def stop(self):
        for i in range(len(self.models)):
            self.queue.put(None)

    def runOne(self, queries):
        self.queue.put(queries)

    def runOneAsync(self, model, queue):
        batch = queue.get()

        while batch is not None:
            responses = []
            for query in batch:
                inputs = self.loader.get(query.index)
                _runOne(model, self.constants, inputs)

                # The last two args are supposed to be for the result data
                # (it's a C pointer and length). These are then logged by
                # loadgen in certain configurations (for accuracy checking
                # mostly). We don't need that feature so we just skip it.
                responses.append(mlperf_loadgen.QuerySampleResponse(query.id, 0, 0))

            mlperf_loadgen.QuerySamplesComplete(responses)

            batch = queue.get()

    def processLatencies(self, latencies):
        infbench.model.processLatencies(self.benchConfig, latencies)


def mlperfBench(modelSpec, benchConfig):
    """Run the mlperf loadgen version"""

    if benchConfig['inline']:
        print("WARNING: inline does nothing in local mode (it's basically always inline)")
    if benchConfig['actors']:
        print("WARNING: useActors does nothing in local mode")

    gpuType = util.getGpuType()

    loader, models = _getHandlers(modelSpec)
    constants = models[0].getConstants(modelSpec.modelPath.parent)

    settings = models[0].getMlPerfCfg(gpuType, testing=benchConfig['testing'])

    runner = mlperfRunner(loader, constants, models, benchConfig)
    runner.start()

    try:
        sut = mlperf_loadgen.ConstructSUT(
            runner.runOne, infbench.model.flushQueries, runner.processLatencies)

        qsl = mlperf_loadgen.ConstructQSL(
            loader.ndata, infbench.model.mlperfNquery, loader.preLoad, loader.unLoad)

        mlperf_loadgen.StartTest(sut, qsl, settings)

        mlperf_loadgen.DestroyQSL(qsl)
        mlperf_loadgen.DestroySUT(sut)
    finally:
        runner.stop()

    infbench.model.reportMlPerf()

# =============================================================================
# Server Mode
# =============================================================================


def serveRequests(benchConfig):
    raise ValueError("Local does not support server mode right now")
