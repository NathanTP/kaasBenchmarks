import ray
import ray.util.queue
import infbench
import threading

import mlperf_loadgen

config = {}
models = {}

def configure(cfg):
    """Must include dataDir and modelDir fields (pathlib paths)"""
    global config
    global models

    config = cfg

    models = {
            "superRes" : {
                    "loader" : infbench.dataset.superResLoader,
                    "modelClass" : infbench.model.superRes,
                    "modelPath" : config['modelDir'] / "super_resolution.onnx"
                }
            }


@ray.remote
def pre(modelName, batch):
    results = []
    for data in batch:
        results.append(models[modelName]['modelClass'].pre(data))

    # We work with batches of inputs and need to return batched results. Since
    # each call to pre returns a tuple, we need to re-arrange into a tuple of
    # lists instead of a list of tuples, zip(*X) is like the inverse of zip.
    # [ (a0, b0), (a1, b1) ] -> ([a0, a1], [b0, b1])
    return tuple(zip(*results))


@ray.remote(num_gpus=1)
def run(modelName, modelBuf, batch):
    results = []
    for data in batch:
        model = models[modelName]['modelClass'](modelBuf)
        results.append(model.run(data))

    return results 


@ray.remote
# def post(modelName, batchRefs, completionQ=None, queryIds=None):
def post(modelName, *batch, completionQ=None, queryIds=None):
    # This works on batches of inputs (each input is a list) zip them up into
    # tuples representing individual queries:
    # batch: [ [preA0, preA1], [preB0, preB1], [run0, run1] ]
    # data: (preA0, preB0, run0)
    # Return: [res0, res1, res2, ...]

    results = []
    for data in zip(*batch):
        results.append(models[modelName]['modelClass'].post(data))

    # In mlperf mode, we need to asynchronously report completions to a worker
    # through this queue. Otherwise we can return a ray future.
    if completionQ is not None:
        completionQ.put((results, queryIds))
        return None
    else:
        return results


@ray.remote(num_gpus=1)
def runInline(modelName, modelBuf, batch, completionQ=None, queryIds=None):
    """Run model with inlined pre and post-processing"""
    modelSpec = models[modelName]

    model = modelSpec['modelClass'](modelBuf)

    results = []
    for data in batch:
        preOut = model.pre(data)

        runIn = preOut[model.runMap]
        runOut = model.run(runIn)

        postIn = [ preOut[i] for i in model.postMap ] + [runOut]
        results.append(model.post(postIn))

    if completionQ is not None:
        completionQ.put((results, queryIds))
        return None
    else:
        return results


def runN(modelName, modelBuf, inputs, inline=False, completionQ=None, queryIds=None):
    """Issue one query asynchronously to ray, returns a future. inline will run
       all data processing steps in the same function as the model."""
    modelSpec = models[modelName]
    mClass = modelSpec['modelClass']

    if inline:
        if completionQ is not None:
            runInline.options(num_returns=mClass.nOutPost).remote(
                    modelName, modelBuf, [inputs], completionQ=completionQ, queryIds=queryIds)
            postOut = None
        else:
            postOut = runInline.options(num_returns=mClass.nOutPost).remote(
                                modelName, modelBuf, [inputs])
    else:
        preOut = pre.options(num_returns=mClass.nOutPre).remote(modelName, [inputs])

        modOut = run.remote(modelName, modelBuf, preOut[mClass.runMap])

        postInp = [ preOut[i] for i in mClass.postMap ] + [modOut]
        if completionQ is not None:
            post.options(num_returns=mClass.nOutPost).remote(
                    modelName, *postInp, completionQ=completionQ, queryIds=queryIds)
            postOut = None
        else:
            postOut = post.options(num_returns=mClass.nOutPost).remote(modelName, *postInp)

    return postOut


def oneShot(modelName, inline=False):
    """Single invocation of the model. This test assumes you have compiled the
    model at least once (the .so is available in the cache)."""
    ray.init()

    modelSpec = models[modelName]
    modelBuf = infbench.model.readModelBuf(modelSpec['modelPath'])
    loader = modelSpec['loader'](config['dataDir'])
    inp = loader.get(0)

    postOut = runN(modelName, modelBuf, [inp], inline=inline)
    return ray.get(postOut)[0]


#==============================================================================
# MLPERF INFERENCE STUFF
#==============================================================================


def handleCompletion(queue):
    """Handle responses from mlperf model serving functions. Each response is a
    list of query IDs that are now complete. To signal completion, the user
    must push an integer to the queue representing the total number of
    responses to expect (including all those already completed).
    
    Note: For now we ignore return values, the client does not care about the
    result of the prediction in this benchmark."""

    # number of batches we've received.
    ncomplete = 0

    # Once set, this is the total number of responses we're expecting
    targetComplete = None
    wrapItUp = False
    while not wrapItUp or ncomplete != targetComplete:
        # One batch at a time
        resps, qids = queue.get()

        # if ncomplete % 10 == 0:
        #     print("Query {} Finished".format(ncomplete))

        if isinstance(resps, int):
            targetComplete = resps

            if ncomplete == targetComplete:
                break
            else:
                wrapItUp = True
        else:
            responses = []
            for i in qids:
                responses.append(mlperf_loadgen.QuerySampleResponse(i, 0, 0))
            mlperf_loadgen.QuerySamplesComplete(responses)
            ncomplete += 1


class mlperfRunner():
    def __init__(self, modelName, inline=False):
        self.modelName = modelName
        self.modelSpec = models[modelName]
        self.modelBuf = infbench.model.readModelBuf(self.modelSpec['modelPath'])
        self.loader = self.modelSpec['loader'](config['dataDir'])
        self.inline = inline

        # Total number of queries issued 
        self.nIssued = 0


    def start(self, preWarm=True):
        self.completionQueue = ray.util.queue.Queue()
        self.completionHandler = threading.Thread(
                target=handleCompletion, args=[self.completionQueue])
        self.completionHandler.start()

        # This is very important for Ray because the cold start is like 1s and
        # mlperf is based on SLOs which we violate immediately.
        if preWarm:
            inputs = [self.loader.get(0)]
            res = runN(self.modelName, self.modelBuf, inputs, inline=self.inline)
            ray.get(res)

    def runBatch(self, queryBatch):
        inputs = []
        qids = []
        for q in queryBatch:
            inputs.append(self.loader.get(q.index))
            qids.append(q.id)

        runN(self.modelName, self.modelBuf, inputs, inline=self.inline,
                completionQ=self.completionQueue, queryIds=qids)

        self.nIssued += len(queryBatch)


    def stop(self):
        self.completionQueue.put((self.nIssued, None))
        print("Waiting for completion handler to finish")
        self.completionHandler.join()


def mlperfBench(modelName, testing=False, inline=False):
    """Run the mlperf loadgen version"""
    modelSpec = models[modelName]
    settings = modelSpec['modelClass'].getMlPerfCfg(testing=testing)
    loader = modelSpec['loader'](config['dataDir'])

    runner = mlperfRunner(modelName, inline)

    ray.init()
    runner.start(preWarm=True)
    print("Starting MLPerf Benchmark:")
    sut = mlperf_loadgen.ConstructSUT(
        runner.runBatch, infbench.model.flushQueries, infbench.model.processLatencies)

    qsl = mlperf_loadgen.ConstructQSL(
        loader.ndata, infbench.model.mlperfNquery, loader.preload, loader.unload)

    mlperf_loadgen.StartTest(sut, qsl, settings)
    mlperf_loadgen.DestroyQSL(qsl)
    mlperf_loadgen.DestroySUT(sut)

    runner.stop()

    infbench.model.reportMlPerf()
