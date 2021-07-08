import ray
import ray.util.queue
import infbench
import threading
import time
import numpy as np
import os

import mlperf_loadgen


# Each phase (pre,run,post) accepts and returns multiple values. Each value is
# itself a list/tuple representing the batch. This facilitates ray.get()'ing a
# subset of the inputs:
# [ [A0, A1], [B0, B1], ... ]
#
# Inside functions, it is often more convenient to use the transpose of this (a
# batch of tuples of inputs):
# [ (A0, B0), (A1, B1), ... ]
def _transposeBatch(batch):
    return list(zip(*batch))

# All steps (pre/run/post) take in multiple arguments as described in
# _transposeBatch. If we passed a list of futures, we would need to ray.get()
# each one seperately. This would prevent Ray from doing full lazy evaluation,
# it would instantiate a million functions each waiting on ray.get(), wasting a
# ton of resources and eventually crashing. Instead, we pass each input
# directly as an argument using the *batch syntax (this groups remaining
# function arguments into a list). This way Ray waits until all inputs are
# ready before instantiating the function.


def _unMarshalArgs(argMap, args):
    """Due to Ray's requirement that all references be passed as arguments, we
    are forced to marshal all variable-length arguments into a single list.
    This unMarshals it back into constants and batched inputs."""
    if argMap.const is None:
        return ([], args)
    else:
        nConst = len(argMap.const)
        constants = list(args[:nConst])
        inputs = list(args[nConst:])
        return (constants, inputs)


@ray.remote
def pre(modelSpec, *batch):
    mClass = modelSpec['modelClass']
    constants, batch = _unMarshalArgs(mClass.preMap, batch)

    batch = _transposeBatch(batch)

    results = []
    for data in batch:
        results.append(mClass.pre(constants + list(data)))

    # If there are multiple returns, ray will return a tuple. If there is only
    # one return, it will return it as a reference to a tuple so we have to
    # unpack it here before returning.
    if len(results[0]) == 1:
        return _transposeBatch(results)[0]
    else:
        return _transposeBatch(results)


# Not sure how to have a truly per-worker cache, but this dict maps PID to the initialized model (if any).
# From what I can tell, Ray will create a pool of processes for each unique
# task. Each task will get its own pool. Since these require a GPU, I would not
# expect the pool to exceed 2 and I would expect ray to kill workers when more
# than two unique tasks require a GPU.
modelCache = {}


@ray.remote(num_gpus=1)
def run(modelSpec, modelBuf, *batch, completionQ=None, queryIds=None, cacheModel=False):
    mClass = modelSpec['modelClass']
    constants, batch = _unMarshalArgs(mClass.runMap, batch)

    batch = _transposeBatch(batch)
    results = []

    if cacheModel:
        pid = os.getpid()
        if pid in modelCache:
            model = modelCache[pid]
        else:
            model = modelSpec['modelClass'](modelBuf)
            modelCache[pid] = model
    else:
        model = modelSpec['modelClass'](modelBuf)

    for data in batch:
        results.append(model.run(constants + list(data)))
        # results.append(model.run(data))

    results = _transposeBatch(results)

    if completionQ is not None:
        completionQ.put((results, queryIds))

    if len(results) == 1:
        return results[0]
    else:
        return results


@ray.remote
def post(modelSpec, *batch, completionQ=None, queryIds=None):
    mClass = modelSpec['modelClass']
    constants, batch = _unMarshalArgs(mClass.runMap, batch)

    batch = _transposeBatch(batch)

    results = []
    for data in batch:
        results.append(modelSpec['modelClass'].post(constants + list(data)))

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
def runInline(modelSpec, modelBuf, *refs, completionQ=None, queryIds=None):
    """Run model with inlined pre and post-processing"""
    model = modelSpec['modelClass'](modelBuf)

    constants = refs[:model.nConst]
    batch = refs[model.nConst:]

    results = []
    batch = _transposeBatch(batch)
    for inputs in batch:
        preInp = _getInputs(model.preMap, const=constants, inp=inputs)
        preOut = model.pre(preInp)

        runInp = _getInputs(model.runMap, const=constants, inp=inputs, pre=preOut)
        modOut = model.run(runInp)

        if model.noPost:
            postOut = modOut
        else:
            postInp = _getInputs(model.postMap, const=constants, inp=inputs, pre=preOut, run=modOut)
            postOut = model.post(postInp)

        results.append(postOut)

    results = _transposeBatch(results)
    if completionQ is not None:
        completionQ.put((results, queryIds))

    return results


def _getInputs(maps, const=None, inp=None, pre=None, run=None):
    inputs = []
    for (argMap, data) in zip(maps, [const, inp, pre, run]):
        if argMap is not None:
            assert data is not None
            inputs.extend([data[i] for i in argMap])
    return inputs


def _runBatch(modelSpec, specRef, modelBuf, constRefs, batch, inline=False, completionQ=None, queryIds=None, cacheModel=False):
    """Issue one query asynchronously to ray, returns a future. inline will run
       all data processing steps in the same function as the model."""
    mClass = modelSpec['modelClass']

    if inline:
        # We can't pass lists of references to ray functions because ray can't
        # statically determine the dataflow. All refs have to be first-class
        # arguments so we pack them all into a list and then expand it with
        # *varArgs
        varArgs = batch
        if constRefs is None:
            varArgs = batch
        else:
            varArgs = list(constRefs) + batch

        if completionQ is not None:
            runInline.options(num_returns=mClass.nOutPost).remote(
                    specRef, modelBuf, *varArgs, completionQ=completionQ, queryIds=queryIds)
            postOut = None
        else:
            postOut = runInline.options(num_returns=mClass.nOutPost).remote(
                                specRef, modelBuf, *varArgs)
    else:
        # Pre
        preInp = _getInputs(mClass.preMap, const=constRefs, inp=batch)

        preOut = pre.options(num_returns=mClass.nOutPre).remote(specRef, *preInp)
        if mClass.nOutPre == 1:
            preOut = [preOut]

        # Run
        runInp = _getInputs(mClass.runMap, const=constRefs, inp=batch, pre=preOut)
        if completionQ is not None and mClass.noPost:
            runOut = run.options(num_returns=mClass.nOutRun).remote(
                    specRef, modelBuf, *runInp, completionQ=completionQ, queryIds=queryIds, cacheModel=cacheModel)
        else:
            runOut = run.options(num_returns=mClass.nOutRun).remote(specRef, modelBuf, *runInp, cacheModel=cacheModel)

        if mClass.nOutRun == 1:
            runOut = [runOut]

        # Post
        if mClass.noPost:
            postOut = runOut
        else:
            postInp = _getInputs(mClass.postMap, const=constRefs, inp=batch, pre=preOut, run=runOut)
            postOut = post.options(num_returns=mClass.nOutPost).remote(
                    specRef, *postInp, completionQ=completionQ, queryIds=queryIds)

            if mClass.nOutPost == 1:
                postOut = [postOut]

    return postOut


def nShot(modelSpec, n, inline=False):
    ray.init()

    specRef = ray.put(modelSpec)

    modelBuf = infbench.model.readModelBuf(modelSpec['modelPath'])
    loader = modelSpec['loader'](modelSpec['dataDir'])

    loader.preLoad(range(min(n, loader.ndata)))
    constants = modelSpec['modelClass'].getConstants(modelSpec['modelPath'].parent)
    if constants is None:
        constRefs = None
    else:
        constRefs = []
        for const in constants:
            constRefs.append(ray.put(const))

    times = []
    accuracies = []
    results = []
    for i in range(n):
        idx = i % loader.ndata
        inp = loader.get(idx)

        batch = _transposeBatch([inp])

        start = time.time()
        res = _runBatch(modelSpec, specRef, modelBuf, constRefs, batch, inline=inline)
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
        print("Accuracy = ", sum([int(res) for res in accuracies]) / n)
    else:
        print("Accuracy checking not supported by this dataset")

    return results


# =============================================================================
# MLPERF INFERENCE STUFF
# =============================================================================


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
    def __init__(self, modelSpec, loader, constantRefs, inline=False):
        self.modelSpec = modelSpec
        self.modelBuf = infbench.model.readModelBuf(self.modelSpec['modelPath'])
        self.loader = loader
        self.constants = constantRefs
        self.inline = inline

        # Total number of queries issued
        self.nIssued = 0

        self.specRef = ray.put(self.modelSpec)

    def start(self, preWarm=True):
        self.completionQueue = ray.util.queue.Queue()
        self.completionHandler = threading.Thread(
                target=handleCompletion, args=[self.completionQueue])
        self.completionHandler.start()

        # This is very important for Ray because the cold start is multiple
        # seconds and mlperf is based on SLOs which we violate immediately.
        if preWarm:
            self.loader.preLoad([0])
            inputs = [self.loader.get(0)]
            res = _runBatch(self.modelSpec, self.specRef, self.modelBuf,
                            self.constants, inputs, inline=self.inline)
            ray.get(res)

    def runBatch(self, queryBatch):
        inputs = []
        qids = []
        for q in queryBatch:
            inputs.append(self.loader.get(q.index))
            qids.append(q.id)

        _runBatch(self.modelSpec, self.specRef, self.modelBuf,
                  self.constants, inputs, inline=self.inline,
                  completionQ=self.completionQueue, queryIds=qids,
                  cacheModel=True)

        self.nIssued += len(queryBatch)

    def stop(self):
        self.completionQueue.put((self.nIssued, None))
        print("Waiting for completion handler to finish")
        self.completionHandler.join()


def mlperfBench(modelSpec, testing=False, inline=False):
    """Run the mlperf loadgen version"""
    ray.init()

    settings = modelSpec['modelClass'].getMlPerfCfg(testing=testing)
    loader = modelSpec['loader'](modelSpec['dataDir'])

    constants = modelSpec['modelClass'].getConstants(modelSpec['modelPath'].parent)
    if constants is None:
        constRefs = None
    else:
        constRefs = []
        for const in constants:
            constRefs.append(ray.put(const))

    runner = mlperfRunner(modelSpec, loader, constRefs, inline=inline)

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
