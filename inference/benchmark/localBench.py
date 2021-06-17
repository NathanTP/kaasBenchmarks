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

    modelSpecs = {
            "superRes" : {
                    "loader"     : infbench.dataset.superResLoader,
                    "modelPath"  : config['modelDir'] / "super_resolution.onnx",
                    "modelClass" : infbench.model.superRes
                },
            "resnet50" : {
                    "loader"     : infbench.dataset.imageNetLoader,
                    "modelPath"  : config['modelDir'] / "resnet50.onnx",
                    "modelClass" : infbench.model.resnet50
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


def _runOne(model, inputs):
    preInp = [ inputs[i] for i in model.preMap ]
    preOut = model.pre(preInp)

    runInp = [ preOut[i] for i in model.runMap ]
    modOut = model.run(runInp)

    if model.noPost:
        postOut = modOut
    else:
        postInp = [ preOut[i] for i in model.postMap ] + modOut
        postOut = model.post(postInp)

    return postOut


def nShot(modelName, n):
    modelSpec = modelSpecs[modelName]
    loader, models = _getHandlers(modelSpec)

    loader.preLoad(list(range( min(n, loader.ndata) )))
    model = models[0]

    times = []
    accuracies = []
    results = []
    for i in range(n):
        idx = i % loader.ndata
        inputs = loader.get(idx)

        start = time.time()

        result = _runOne(model, inputs)

        times.append(time.time() - start)
        results.append(result)
        accuracies.append(loader.check(result, idx))

    print("Average latency: ")
    print(np.mean(times))
    print("Median latency: ")
    print(np.percentile(times, 50))
    print("99 percentile latency: ")
    print(np.percentile(times, 99))

    print("Accuracy = ", sum([ int(res) for res in accuracies ]) / n)

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

        mlperf_loadgen.StartTest(sut, qsl, settings)
        mlperf_loadgen.DestroyQSL(qsl)
        mlperf_loadgen.DestroySUT(sut)
    finally:
        runner.stop()

    infbench.model.reportMlPerf()
