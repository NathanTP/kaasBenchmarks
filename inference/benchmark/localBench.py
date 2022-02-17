import infbench
import util

import mlperf_loadgen
from gpuinfo import GPUInfo
from pprint import pprint
import json
import pathlib

import kaas
import kaas.local
from kaas import profiling

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


# We need every key in the obj store to be unique
nextId = 0


def getNextKey(n=None):
    global nextId

    if n is None:
        newId = nextId
        nextId += 1
        return newId
    else:
        newIds = [nextId + (i-1) for i in range(n)]
        nextId += n
        return newIds


def pre(model, kv, inputs, constants):
    preInp = kv.get(util.packInputs(model.preMap, const=constants, inp=inputs))
    outVals = model.pre(preInp)
    outKeys = getNextKey(len(outVals))
    for k, v in zip(outKeys, outVals):
        kv.put(k, v)

    return outKeys


def post(model, kv, constants, inputs, preOut, runOut):
    postInp = kv.get(util.packInputs(model.postMap, const=constants, inp=inputs, pre=preOut, run=runOut))
    outVals = model.post(postInp)
    outKeys = getNextKey(len(outVals))
    for k, v in zip(outKeys, outVals):
        kv.put(k, v)

    return outKeys


def runKaas(model, kv, constants, inputs, preOut, stats=None):
    """Run a kaas model. inputs and preOut are literal values, constants should
    be keys in kaasCtx.kv for the constants"""
    global kaasNextId

    outKeys = getNextKey(model.nOutRun)
    runInp = util.packInputs(model.runMap, const=constants, inp=inputs, pre=preOut)
    packedReq = model.run(runInp, outKeys=outKeys)

    kaas.local.invoke(packedReq, kv, stats=stats)

    return outKeys


def runNative(model, kv, constants, inputs, preOut, stats=None):
    runInpKeys = util.packInputs(model.runMap, const=constants, inp=inputs, pre=preOut)
    runInp = [kv.get(k) for k in runInpKeys]

    runOuts = model.run(runInp)

    runKeys = getNextKey(len(runOuts))
    for k, v in zip(runKeys, runOuts):
        kv.put(k, v)

    return runKeys


def _runOne(model, constKeys, inpKeys, kv, mode='direct', stats=None):
    if model.noPre:
        preOutKeys = inpKeys
    else:
        with profiling.timer("t_pre", stats):
            preOutKeys = pre(model, kv, inpKeys, constKeys)

    with profiling.timer("t_run", stats):
        if mode == 'kaas':
            runOutKeys = runKaas(model, kv, constKeys, inpKeys, preOutKeys, stats=stats)
        else:
            runOutKeys = runNative(model, kv, constKeys, inpKeys, preOutKeys, stats=stats)

    if model.noPost:
        postOutKeys = runOutKeys
    else:
        with profiling.timer("t_post", stats):
            postOutKeys = post(model, kv, constKeys, inpKeys, preOutKeys, runOutKeys)

    # Clean up intermediate data so we don't fill up memory
    if not model.noPre:
        kv.delete(preOutKeys)
    if not model.noPost:
        kv.delete(runOutKeys)

    return postOutKeys


def deepProfile(modelSpec, benchConfig, reportPath='results.json'):
    cold = benchConfig['forceCold']

    if modelSpec.modelType == 'kaas':
        mode = 'kaas'
        kaas.local.init()
    else:
        mode = 'direct'

    kv = kaas.local.LocalKV(serialize=False)
    loader = modelSpec.loader(modelSpec.dataDir)
    loader.preLoad([0])

    constants = modelSpec.modelClass.getConstants(modelSpec.modelPath.parent)
    if constants is not None:
        constKeys = getNextKey(len(constants))
        for k, v in zip(constKeys, constants):
            kv.put(k, v)
    else:
        constKeys = []

    runConstKeys = []
    if modelSpec.modelClass.runMap.const is not None:
        for idx in modelSpec.modelClass.runMap.const:
            runConstKeys.append(constKeys[idx])

    if not isinstance(reportPath, pathlib.Path):
        reportPath = pathlib.Path(reportPath)

    coldStats = profiling.profCollection()
    warmStats = profiling.profCollection()

    if infbench.getNGpu() != 1:
        raise ValueError("Deep Profile should be run with only one GPU (try setting the CUDA_VISIBLE_DEVICES environment variable)")

    inputs = loader.get(0)
    inputKeys = getNextKey(len(inputs))
    for k, v in zip(inputKeys, inputs):
        kv.put(k, v)

    # Cold Start
    with profiling.cudaProfile(enable=cold):
        model = modelSpec.getModelInstance(constRefs=runConstKeys, backend='local')
        _runOne(model, constKeys, inputKeys, kv, mode=mode, stats=coldStats)

    # Warm Start
    with profiling.timer("t_e2e", warmStats):
        with profiling.cudaProfile(enable=not cold):
            _runOne(model, constKeys, inputKeys, kv, mode=mode, stats=warmStats)

    fullReport = {}
    fullReport['config'] = benchConfig

    fullReport['coldMetrics'] = coldStats.report()
    fullReport['warmMetrics'] = warmStats.report()

    print("Cold Stats: ")
    util.analyzeStats(fullReport['coldMetrics'])

    print("\nWarm Stats: ")
    util.analyzeStats(fullReport['warmMetrics'])

    if reportPath.exists():
        reportPath.unlink()

    print("Saving results to: ", reportPath)
    with open(reportPath, 'w') as f:
        json.dump(fullReport, f)


def nShot(modelSpec, n, benchConfig, reportPath="results.json"):
    stats = profiling.profCollection()

    if modelSpec.modelType == 'kaas':
        mode = 'kaas'
        kaas.local.init()
    else:
        mode = 'direct'

    kv = kaas.local.LocalKV(serialize=False)
    loader = modelSpec.loader(modelSpec.dataDir)

    constants = modelSpec.modelClass.getConstants(modelSpec.modelPath.parent)
    if constants is not None:
        constKeys = getNextKey(len(constants))
        for k, v in zip(constKeys, constants):
            kv.put(k, v)
    else:
        constKeys = []

    runConstKeys = []
    if modelSpec.modelClass.runMap.const is not None:
        for idx in modelSpec.modelClass.runMap.const:
            runConstKeys.append(constKeys[idx])

    model = modelSpec.getModelInstance(constRefs=runConstKeys, backend='local')

    loader.preLoad(range(min(max(n, infbench.getNGpu()*2), loader.ndata)))

    # Cold starts
    coldInputs = loader.get(0)
    coldKeys = getNextKey(len(coldInputs))
    for k, v in zip(coldKeys, coldInputs):
        kv.put(k, v)

    for i in range(infbench.getNGpu()):
        _runOne(model, constKeys, coldKeys, kv, mode=mode, stats=stats)

    # coldReport = stats.report()
    stats.reset()

    accuracies = []
    results = []
    for i in range(n):
        idx = i % loader.ndata
        inputs = loader.get(idx)
        inpKeys = getNextKey(len(inputs))
        for k, v in zip(inpKeys, inputs):
            kv.put(k, v)

        cuda.start_profiler()
        with profiling.timer("t_e2e", stats):
            resultKeys = _runOne(model, constKeys, inpKeys, kv, mode=mode, stats=stats)
        cuda.stop_profiler()

        resultVals = [kv.get(k) for k in resultKeys]
        results.append(resultVals)

        if loader.checkAvailable:
            accuracies.append(loader.check(resultVals, idx))

    if loader.checkAvailable:
        print("Accuracy = ", sum([int(res) for res in accuracies]) / n)
    else:
        print("Dataset does not support accuracy calculation")

    # make sure kaasHandle stats are fully up to date
    report = stats.report()
    print("Detailed Profile: ")
    util.analyzeStats(report)

    print("E2E Results:")
    pprint({(k, v) for (k, v) in report['t_e2e'].items() if k != "events"})

    # if not isinstance(reportPath, pathlib.Path):
    #     reportPath = pathlib.Path(reportPath).resolve()
    #
    # print("Saving results to: ", reportPath)
    # if reportPath.exists():
    #     with open(reportPath, 'r') as f:
    #         fullReport = json.load(f)
    # else:
    #     fullReport = []
    #
    # record = {
    #     "config": benchConfig,
    #     "metrics": report
    # }
    # fullReport.append(record)
    #
    # with open(reportPath, 'w') as f:
    #     json.dump(fullReport, f)

    return results


# =============================================================================
# MLPERF INFERENCE STUFF
# =============================================================================

class mlperfRunner():

    def __init__(self, loader, constKeys, model, kv, benchConfig):
        self.loader = loader
        self.kv = kv
        self.model = model
        self.constKeys = constKeys
        self.benchConfig = benchConfig

    def runOne(self, batch):
        responses = []
        for query in batch:
            inputs = self.loader.get(query.index)
            inpKeys = getNextKey(len(inputs))
            for k, v in zip(inpKeys, inputs):
                self.kv.put(k, v)

            outKeys = _runOne(self.model, self.constKeys, inpKeys, self.kv)

            # Some models have big objects, best to clear out so we don't
            # run out of memory
            self.kv.delete(inpKeys + outKeys)

            # The last two args are supposed to be for the result data
            # (it's a C pointer and length). These are then logged by
            # loadgen in certain configurations (for accuracy checking
            # mostly). We don't need that feature so we just skip it.
            responses.append(mlperf_loadgen.QuerySampleResponse(query.id, 0, 0))

        mlperf_loadgen.QuerySamplesComplete(responses)

    def processLatencies(self, latencies):
        self.latMetrics = infbench.processLatencies(self.benchConfig, latencies)


def mlperfBench(modelSpec, benchConfig):
    """Run the mlperf loadgen version"""

    if modelSpec.modelType == 'kaas':
        kaas.local.init()

    kv = kaas.local.LocalKV(serialize=False)
    loader = modelSpec.loader(modelSpec.dataDir)

    constants = modelSpec.modelClass.getConstants(modelSpec.modelPath.parent)
    if constants is not None:
        constKeys = getNextKey(len(constants))
        for k, v in zip(constKeys, constants):
            kv.put(k, v)
    else:
        constKeys = []

    runConstKeys = []
    if modelSpec.modelClass.runMap.const is not None:
        for idx in modelSpec.modelClass.runMap.const:
            runConstKeys.append(constKeys[idx])

    model = modelSpec.getModelInstance(constRefs=runConstKeys, backend='local')

    gpuType = infbench.getGpuType()

    runner = mlperfRunner(loader, constKeys, model, kv, benchConfig)

    sut = mlperf_loadgen.ConstructSUT(
        runner.runOne, infbench.model.flushQueries, runner.processLatencies)

    qsl = mlperf_loadgen.ConstructQSL(
        loader.ndata, infbench.model.mlperfNquery, loader.preLoad, loader.unLoad)

    settings = model.getMlPerfCfg(gpuType, benchConfig)
    mlperf_loadgen.StartTest(sut, qsl, settings)

    mlperf_loadgen.DestroyQSL(qsl)
    mlperf_loadgen.DestroySUT(sut)

    print("\nResults:")
    mlPerfMetrics, valid = infbench.parseMlPerf('mlperf_log_')
    print(mlPerfMetrics.report())

    # print("\nStats:")
    # report = warmStats.report()
    # util.analyzeStats(report)
    #
    # infbench.model.saveReport(mlPerfMetrics, benchConfig, 'results.json')

# =============================================================================
# Server Mode
# =============================================================================


def serveRequests(benchConfig):
    raise ValueError("Local does not support server mode right now")
