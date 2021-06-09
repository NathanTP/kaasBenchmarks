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


def nShot(modelName, n):
    modelSpec = modelSpecs[modelName]
    loader, dataProc, model = _getHandlers(modelSpec)

    inp = loader.get(0)

    stops = []
    for i in range(n):
        inp = loader.get(i % loader.ndata)

        start = time.time()

        preOut = dataProc.pre([inp])
        modOut = model.run(preOut[model.inpMap[0]])
        postInp = [ preOut[i] for i in dataProc.postMap ] + [modOut]
        postOut = dataProc.post(postInp)

        stops.append(time.time() - start)
        print("Took: ", stops[-1])

    print("Average latency: ")
    print(np.mean(stops))
    print("Median latency: ")
    print(np.percentile(stops, 50))
    print("99 percentile latency: ")
    print(np.percentile(stops, 99))

def oneShot(modelName):
    modelSpec = modelSpecs[modelName]
    loader, models = _getHandlers(modelSpec)
    model = models[0]

    inp = loader.get(0)

    preOut = model.pre([inp])

    modOut = model.run(preOut[model.runMap])

    postInp = [ preOut[i] for i in model.postMap ] + [modOut]
    postOut = model.post(postInp)
    return postOut


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
                inp = self.loader.get(query.index)
                preOut = model.pre([inp])
                modOut = model.run(preOut[model.runMap])
                postInp = [ preOut[i] for i in model.postMap ] + [modOut]
                postOut = model.post(postInp)

                # XXX I really don't know what the last two args are for. The first
                # is the memory address of the response, the second is the length
                # (in bytes) of the response. I don't know what mlperf does with
                # these, hopefully leaving them out doesn't break anything. You can
                # see the mlperf vision examples to see an example of actually
                # providing them.
                responses.append(mlperf_loadgen.QuerySampleResponse(query.id, 0, 0))

            mlperf_loadgen.QuerySamplesComplete(responses)

            batch = queue.get()


def mlperfBench(modelName):
    """Run the mlperf loadgen version"""

    modelSpec = modelSpecs[modelName]
    loader, models = _getHandlers(modelSpec)

    settings = models[0].getMlPerfCfg(testing=True)

    runner = mlperfRunner(loader, models)
    runner.start()
    try:
        print("Starting MLPerf Benchmark:")
        sut = mlperf_loadgen.ConstructSUT(
            runner.runOne, infbench.model.flushQueries, infbench.model.processLatencies)

        qsl = mlperf_loadgen.ConstructQSL(
            loader.ndata, infbench.model.mlperfNquery, loader.preload, loader.unload)

        mlperf_loadgen.StartTest(sut, qsl, settings)
        mlperf_loadgen.DestroyQSL(qsl)
        mlperf_loadgen.DestroySUT(sut)
    finally:
        runner.stop()

    infbench.model.reportMlPerf()
