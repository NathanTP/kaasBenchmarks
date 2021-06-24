import ray
import ray.util.queue
import infbench
import threading
import time
import numpy as np

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
                },
            "resnet50" : {
                    "loader"     : infbench.dataset.imageNetLoader,
                    "modelPath"  : config['modelDir'] / "resnet50.onnx",
                    "modelClass" : infbench.model.resnet50
                },
            "ssdMobilenet" : {
                    "loader"     : infbench.dataset.cocoLoader,
                    "modelPath"  : config['modelDir'] / "ssdMobilenet.so",
                    "modelClass" : infbench.model.ssdMobilenet
                }
            }


# Each phase (pre,run,post) accepts and returns multiple values. Each value is
# itself a list/tuple representing the batch. This facilitates ray.get()'ing a
# subset of the inputs:
# [ [A0, A1], [B0, B1], ... ]
#
# Inside functions, it is often more convenient to use the transpose of this (a
# batch of tuples of inputs):
# [ (A0, B0), (A1, B1), ... ]
def _transposeBatch(batch):
    return tuple(zip(*batch))

# All steps (pre/run/post) take in multiple arguments as described in
# _transposeBatch. If we passed a list of futures, we would need to ray.get()
# each one seperately. This would prevent Ray from doing full lazy evaluation,
# it would instantiate a million functions each waiting on ray.get(), wasting a
# ton of resources and eventually crashing. Instead, we pass each input
# directly as an argument using the *batch syntax (this groups remaining
# function arguments into a list). This way Ray waits until all inputs are
# ready before instantiating the function.

@ray.remote
def pre(modelName, *batch):
    batch = _transposeBatch(batch)

    results = []
    for data in batch:
        results.append(models[modelName]['modelClass'].pre(data))

    # If there are multiple returns, ray will return a tuple. If there is only
    # one return, it will return it as a reference to a tuple so we have to
    # unpack it here before returning.
    if len(results[0]) == 1:
        return _transposeBatch(results)[0]
    else:
        return _transposeBatch(results)


@ray.remote(num_gpus=1)
def run(modelName, modelBuf, *batch, completionQ=None, queryIds=None):
    batch = _transposeBatch(batch)
    results = []

    model = models[modelName]['modelClass'](modelBuf)
    for data in batch:
        results.append(model.run(data))

    results = _transposeBatch(results)

    if completionQ is not None:
        completionQ.put((results, queryIds))

    if len(results) == 1:
        return results[0]
    else:
        return results

  
@ray.remote
def post(modelName, *batch, completionQ=None, queryIds=None):
    batch = _transposeBatch(batch)

    results = []
    for data in batch:
        results.append(models[modelName]['modelClass'].post(data))

    results = _transposeBatch(results)

    # In mlperf mode, we need to asynchronously report completions to a worker
    # through this queue. Otherwise we can return a ray future.
    if completionQ is not None:
        completionQ.put((results, queryIds))

    if len(results) == 1:
        return results[0]
    else:
        return results


@ray.remote(num_gpus=1)
def runInline(modelName, modelBuf, batch, completionQ=None, queryIds=None):
    """Run model with inlined pre and post-processing"""
    modelSpec = models[modelName]

    model = modelSpec['modelClass'](modelBuf)

    results = []
    batch = _transposeBatch(batch)
    for inputs in batch:
        preInp = [ inputs[i] for i in model.preMap ]
        preOut = model.pre(preInp)

        runInp = [ preOut[i] for i in model.runMap ]
        modOut = model.run(runInp)

        if model.noPost:
            postOut = modOut
        else:
            postInp = [ preOut[i] for i in model.postMap ] + modOut
            postOut = model.post(postInp)

        results.append(postOut)

    results = _transposeBatch(results)
    if completionQ is not None:
        completionQ.put((results, queryIds))

    return results


def _runBatch(modelName, modelBuf, batch, inline=False, completionQ=None, queryIds=None):
    """Issue one query asynchronously to ray, returns a future. inline will run
       all data processing steps in the same function as the model."""
    modelSpec = models[modelName]
    mClass = modelSpec['modelClass']

    if inline:
        if completionQ is not None:
            runInline.options(num_returns=mClass.nOutPost).remote(
                    modelName, modelBuf, batch, completionQ=completionQ, queryIds=queryIds)
            postOut = None
        else:
            postOut = runInline.options(num_returns=mClass.nOutPost).remote(
                                modelName, modelBuf, batch)
    else:
        # Pre 
        preInp = [ batch[i] for i in mClass.preMap ]
        preOut = pre.options(num_returns=mClass.nOutPre).remote(modelName, *preInp)
        if mClass.nOutPre == 1:
            preOut = [preOut]

        # Run
        runInp = [ preOut[i] for i in mClass.runMap ]
        if completionQ is not None and mClass.noPost:
            runOut = run.options(num_returns=mClass.nOutRun).remote(
                    modelName, modelBuf, *runInp, completionQ=completionQ, queryIds=queryIds)
        else:
            runOut = run.options(num_returns=mClass.nOutRun).remote(modelName, modelBuf, *runInp)

        if mClass.nOutRun == 1:
            runOut = [runOut]

        # Post
        if mClass.noPost:
            postOut = runOut
        else:
            postInp = [ preOut[i] for i in mClass.postMap ] + runOut
            postOut = post.options(num_returns=mClass.nOutPost).remote(
                    modelName, *postInp, completionQ=completionQ, queryIds=queryIds)

            if mClass.nOutPost == 1:
                postOut = [postOut]

    return postOut


def nShot(modelName, n, inline=False):
    ray.init()

    modelSpec = models[modelName]
    modelBuf = infbench.model.readModelBuf(modelSpec['modelPath'])
    loader = modelSpec['loader'](config['dataDir'])

    loader.preLoad(range(min(n, loader.ndata)))

    times = []
    accuracies = []
    results = []
    for i in range(n):
        idx = i % loader.ndata
        inp = loader.get(idx)

        batch = _transposeBatch([inp])

        start = time.time()
        res = _runBatch(modelName, modelBuf, batch, inline=inline)
        res = ray.get(res)
        results.append(res[0])

        times.append(time.time() - start)

        res = _transposeBatch(res)

        # nShot always uses batch size 1, so just check idx 0
        if loader.checkAvailable:
            accuracies.append(loader.check(res[0], idx))

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
        print("Accuracy checking not supported by this dataset")

    return results


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
            # The driver is asking us to wrap up and exit after we've seen
            # 'resps' many responses.
            targetComplete = resps

            if ncomplete == targetComplete:
                break
            else:
                wrapItUp = True
        else:
            # Normal completion message from a worker
            completions = []
            for i in qids:
                completions.append(mlperf_loadgen.QuerySampleResponse(i, 0, 0))
            mlperf_loadgen.QuerySamplesComplete(completions)
            ncomplete += 1


class mlperfRunner():
    def __init__(self, modelName, loader, inline=False):
        self.modelName = modelName
        self.modelSpec = models[modelName]
        self.modelBuf = infbench.model.readModelBuf(self.modelSpec['modelPath'])
        # self.loader = self.modelSpec['loader'](config['dataDir'])
        self.loader = loader 
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
            self.loader.preLoad([0])
            inputs = [self.loader.get(0)]
            res = _runBatch(self.modelName, self.modelBuf, inputs, inline=self.inline)
            ray.get(res)

    def runBatch(self, queryBatch):
        inputs = []
        qids = []
        for q in queryBatch:
            inputs.append(self.loader.get(q.index))
            qids.append(q.id)

        _runBatch(self.modelName, self.modelBuf, inputs, inline=self.inline,
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

    runner = mlperfRunner(modelName, loader, inline=inline)

    ray.init()
    runner.start(preWarm=True)
    sut = mlperf_loadgen.ConstructSUT(
        runner.runBatch, infbench.model.flushQueries, infbench.model.processLatencies)

    qsl = mlperf_loadgen.ConstructQSL(
        loader.ndata, infbench.model.mlperfNquery, loader.preLoad, loader.unLoad)

    mlperf_loadgen.StartTest(sut, qsl, settings)
    mlperf_loadgen.DestroyQSL(qsl)
    mlperf_loadgen.DestroySUT(sut)

    runner.stop()

    infbench.model.reportMlPerf()
