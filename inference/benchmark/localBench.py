import infbench
import mlperf_loadgen
import numpy as np
import time
import threading
import queue
from gpuinfo import GPUInfo

config = {}
modelSpecs = {}


def configure(cfg):
    """Must include dataDir and modelDir fields (pathlib paths)"""
    global config
    global modelSpecs

    config = cfg

    # You must run tools/getModels.py first to get these .so's
    modelSpecs = {
        "superRes" : {
                "loader"     : infbench.dataset.superResLoader,
                "modelPath"  : config['modelDir'] / "superres.so",
                "modelClass" : infbench.model.superRes
            },
        "resnet50" : {
                "loader"     : infbench.dataset.imageNetLoader,
                "modelPath"  : config['modelDir'] / "resnet50.so",
                "modelClass" : infbench.model.resnet50
            },
        "ssdMobilenet" : {
                "loader"     : infbench.dataset.cocoLoader,
                "modelPath"  : config['modelDir'] / "ssdMobilenet.so",
                "modelClass" : infbench.model.ssdMobilenet
            },
        "bert" : {
                "loader"     : infbench.dataset.bertLoader,
                "modelPath"  : config['modelDir'] / 'bert' / "bert.so",
                "modelClass" : infbench.model.bertModel
            }
    }


def _getHandlers(modelSpec):
    loader = modelSpec['loader'](config['dataDir'])

    # Create as many models as we have GPUs to get some concurrency. The local
    # mode doesn't independently scale pre/post/run.
    models = []
    for i in range(len(GPUInfo.check_empty())):
        models.append(modelSpec['modelClass'](modelSpec['modelPath']))

    return (loader, models)


def _getInputs(maps, const=None, inp=None, pre=None, run=None):
    inputs = []
    for (argMap, data) in zip(maps, [const, inp, pre, run]):
        if argMap is not None:
            assert data is not None
            inputs.extend([data[i] for i in argMap])
    return inputs


def _runOne(model, constants, inputs):
    preInp = _getInputs(model.preMap, const=constants, inp=inputs)
    preOut = model.pre(preInp)

    runInp = _getInputs(model.runMap, const=constants, inp=inputs, pre=preOut)
    modOut = model.run(runInp)

    if model.noPost:
        postOut = modOut
    else:
        postInp = _getInputs(model.postMap, const=constants, inp=inputs, pre=preOut, run=modOut)
        postOut = model.post(postInp)

    return postOut


def nShot(modelName, n, inline=False):
    modelSpec = modelSpecs[modelName]
    loader, models = _getHandlers(modelSpec)

    if inline:
        print("WARNING: inline does nothing in local mode (it's basically always inline)")

    loader.preLoad(list(range(min(n, loader.ndata))))
    model = models[0]
    constants = model.getConstants(modelSpec['modelPath'].parent)

    times = []
    accuracies = []
    results = []
    for i in range(n):
        idx = i % loader.ndata
        inputs = loader.get(idx)

        start = time.time()

        result = _runOne(model, constants, inputs)

        times.append(time.time() - start)
        results.append(result)

        if loader.checkAvailable:
            accuracies.append(loader.check(result, idx))

    print("Minimum latency: ")
    print(np.min(times))
    print("Maximum latency: ")
    print(np.max(times))
    print("Average latency: ")
    print(np.mean(times))
    print("Median latency: ")
    print(np.percentile(times, 50))
    print("99 percentile latency: ")
    print(np.percentile(times, 99))

    if loader.checkAvailable:
        print("Accuracy = ", sum([ int(res) for res in accuracies ]) / n)
    else:
        print("Dataset does not support accuracy calculation")

    return results


#==============================================================================
# MLPERF INFERENCE STUFF
#==============================================================================

class mlperfRunner():

    def __init__(self, loader, models):
        self.loader = loader
        self.models = models
        self.queue = queue.SimpleQueue()

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
                result = _runOne(model, inputs)

                # XXX I really don't know what the last two args are for. The first
                # is the memory address of the response, the second is the length
                # (in bytes) of the response. I don't know what mlperf does with
                # these, hopefully leaving them out doesn't break anything. You can
                # see the mlperf vision examples to see an example of actually
                # providing them.
                responses.append(mlperf_loadgen.QuerySampleResponse(query.id, 0, 0))

            mlperf_loadgen.QuerySamplesComplete(responses)

            batch = queue.get()


def mlperfBench(modelName, testing=False, inline=False):
    """Run the mlperf loadgen version"""

    if inline:
        print("WARNING: inline does nothing in local mode (it's basically always inline)")

    modelSpec = modelSpecs[modelName]
    loader, models = _getHandlers(modelSpec)

    settings = models[0].getMlPerfCfg(testing=testing)

    runner = mlperfRunner(loader, models)
    runner.start()
    try:
        sut = mlperf_loadgen.ConstructSUT(
            runner.runOne, infbench.model.flushQueries, infbench.model.processLatencies)

        qsl = mlperf_loadgen.ConstructQSL(
            loader.ndata, infbench.model.mlperfNquery, loader.preLoad, loader.unLoad)

        start = time.time()
        mlperf_loadgen.StartTest(sut, qsl, settings)
        print("TOOK: ", time.time() - start)
        mlperf_loadgen.DestroyQSL(qsl)
        mlperf_loadgen.DestroySUT(sut)
    finally:
        runner.stop()

    infbench.model.reportMlPerf()
