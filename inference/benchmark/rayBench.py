import ray
import ray.util.queue
import infbench
import threading
import os
import signal
import pathlib
import json
import math

import time
import asyncio

from tornado.ioloop import IOLoop
import zmq
from zmq.eventloop.zmqstream import ZMQStream

import mlperf_loadgen
import libff.kaas.kaasRay as kaasRay

import util

# There are tradeoffs to using asyncio vs thread pools for policies. Asyncio is
# a bit slower for unknown reasons, but it's easier to implement policies so
# we're sticking with it for now
maxOutstanding = 32
# USE_THREADED_POLICY = True
USE_THREADED_POLICY = False
if USE_THREADED_POLICY:
    import policy
    # This is effectively the outstanding request window for policies. It's not
    # clear the impact of more threads, though a very large number will
    # certainly cause problems. Tuning this parameter has been ad-hoc so far.
    # If we have a large backlog of requests, we probably have bigger problems
    # than running an optimal policy.
    policyNThread = 32
else:
    import policy_async as policy

# Prof levels control the level of detail recorded, higher levels may have an
# impact on performance.
# PROF_LEVEL = 1  # minimal performance impact
PROF_LEVEL = 2  # serializes a lot of stuff, really slows down e2e

level1Stats = {
    't_e2e',  # total time for the whole pipeline as observed from the client
    't_model_run',  # Time to run just the model from the perspective of the run task
    't_kaas_generate_req',  # Time to create the KaaS req on the client
    't_register_model',  # One-time setup of the model (loading constants, etc)
    't_init_loader',  # Loader init and preload time
}

level2Stats = {
    't_pre',  # total time to run pre() as observed from the client
    't_run',  # total time to run run() as observed from the client
    't_post'  # total time to run post() as observed from the client
}


# All steps (pre/run/post) take in multiple arguments (even if there's one
# argument, it's passed as a tuple). If we passed a list of futures, we would
# need to ray.get() each one seperately. This would prevent Ray from doing full
# lazy evaluation, it would instantiate a million functions each waiting on
# ray.get(), wasting a ton of resources and eventually crashing. Instead, we
# pass each input directly as an argument using the *batch syntax (this groups
# remaining function arguments into a list). This way Ray waits until all
# inputs are ready before instantiating the function.


def _unMarshalArgs(argMap, args):
    """Due to Ray's requirement that all references be passed as arguments, we
    are forced to marshal all variable-length arguments into a single list.
    This unMarshals it back into constants and batched inputs."""
    if argMap.const is None:
        return ([], list(args))
    else:
        nConst = len(argMap.const)
        constants = list(args[:nConst])
        inputs = list(args[nConst:])
        return (constants, inputs)


def maybeDereference(res):
    # KaaS will place the result directly into the object store and return a
    # reference to it. The router also wraps outputs of run. Other inputs (e.g.
    # from pre()) will already be dereferenced by ray.

    if not isinstance(res, list):
        if isinstance(res, ray._raylet.ObjectRef):
            return ray.get(res)
        else:
            return res
    else:
        for i in range(len(res)):
            if isinstance(res[i], ray._raylet.ObjectRef):
                res[i] = ray.get(res[i])

        return res


@ray.remote
def pre(modelSpec, *inputs):
    mClass = modelSpec.modelClass
    constants, data = _unMarshalArgs(mClass.preMap, inputs)

    res = mClass.pre(constants + list(data))
    if len(res) == 1:
        return res[0]
    else:
        return res


# Not sure how to have a truly per-worker cache, but this dict maps PID to the initialized model (if any).
# From what I can tell, Ray will create a pool of processes for each unique
# task. Each task will get its own pool. Since these require a GPU, I would not
# expect the pool to exceed 2 and I would expect ray to kill workers when more
# than two unique tasks require a GPU.
# We assume that clients can only register one model.
# {pid -> {clientID -> model}}
modelCache = {}


def _run(model, inputs, completionQ, queryId, stats=None):
    """Internal run function"""
    constants, data = _unMarshalArgs(model.runMap, inputs)

    with infbench.timer('t_model_run', stats):
        results = model.run(constants + list(data), stats=stats)

    if completionQ is not None:
        completionQ.put((results, queryId))

    # Ray will interpret the return value as tuple if there are multiple
    # returns, but if there is one return, it will treat it as a scalar.
    if len(results) == 1:
        return results[0]
    else:
        return results


@ray.remote(num_gpus=1)
def runKaasTask(req, queryId=None, completionQ=None):
    results = kaasRay.kaasServeRay(req.toDict())

    if completionQ is not None:
        completionQ.put((results, queryId))

    return results


@ray.remote(num_gpus=1)
def runTask(modelSpec, modelArg, *inputs, completionQ=None, queryId=None, cacheModel=False, clientID=None):
    """Run the request as a Ray task"""
    if cacheModel:
        pid = os.getpid()
        if pid not in modelCache:
            modelCache[pid] = {}

        nodeCache = modelCache[pid]
        if clientID in nodeCache:
            model = nodeCache[clientID]
        else:
            model = modelSpec.modelClass(modelArg)
            nodeCache[clientID] = model
    else:
        model = modelSpec.modelClass(modelArg)

    return _run(model, inputs, completionQ, queryId)


@ray.remote
def post(modelSpec, *inputs, completionQ=None, queryId=None):
    mClass = modelSpec.modelClass
    constants, rawData = _unMarshalArgs(mClass.postMap, inputs)

    # The router actor wraps data in an additional reference
    data = maybeDereference(rawData)

    if modelSpec.modelType == 'kaas':
        data = maybeDereference(data)

    results = modelSpec.modelClass.post(constants + list(data))

    # In mlperf mode, we need to asynchronously report completions to a worker
    # through this queue. Otherwise we can return a ray future.
    if completionQ is not None:
        completionQ.put((results, queryId))

    if len(results) == 1:
        return results[0]
    else:
        return results


@ray.remote(num_gpus=1)
def runInline(modelSpec, modelArg, *refs, completionQ=None, queryId=None):
    """Run model with inlined pre and post-processing"""
    model = modelSpec.modelClass(modelArg)

    constants = refs[:model.nConst]
    inputs = refs[model.nConst:]

    preInp = util.packInputs(model.preMap, const=constants, inp=inputs)
    preOut = model.pre(preInp)

    runInp = util.packInputs(model.runMap, const=constants, inp=inputs, pre=preOut)
    modOut = model.run(runInp)

    if model.noPost:
        postOut = modOut
    else:
        postInp = util.packInputs(model.postMap, const=constants, inp=inputs,
                                  pre=preOut, run=modOut)
        postOut = model.post(postInp)

    if completionQ is not None:
        completionQ.put((postOut, queryId))

    return postOut


@ray.remote(num_gpus=1)
class runActor():
    """A persistent actor for running model requests. Actors will cache models
    as needed and run them natively. It is possible to run out of GPU memory
    with actors since they cache every model they are passed."""
    def __init__(self):
        self.modelCache = {}
        # {clientID -> infbench.profCollection}
        self.stats = {}

    def runNative(self, modelInfo, inputRefs, completionQ=None, queryId=None,
                  cacheModel=False, clientID=None):

        if clientID not in self.stats:
            self.stats[clientID] = infbench.profCollection()

        # The runActor must cache the model, if you wan't to reset, you must
        # kill and restart the actor. cacheModel is kept for consistency with
        # runTask but is ignored here.
        if clientID in self.modelCache:
            model = self.modelCache[clientID]
        else:
            with infbench.timer("t_model_init", self.stats[clientID]):
                with infbench.timer('t_loadInput', self.stats[clientID], final=False):
                    modelSpec = ray.get(modelInfo[0])
                    modelArg = ray.get(modelInfo[1])
                model = modelSpec.modelClass(modelArg)
                self.modelCache[clientID] = model

        with infbench.timer('t_loadInput', self.stats[clientID]):
            inputs = ray.get(inputRefs)

        result = _run(model, inputs, completionQ, queryId, stats=self.stats[clientID])

        return result

    def runKaas(self, req, queryId=None, completionQ=None, clientID=None):
        if clientID not in self.stats:
            self.stats[clientID] = infbench.profCollection()

        with infbench.timer('t_model_run', self.stats[clientID]):
            results = kaasRay.kaasServeRay(req, stats=self.stats[clientID].mod('kaas'))

        if completionQ is not None:
            completionQ.put((results, queryId))

        return results

    def terminate(self):
        ray.actor.exit_actor()

    def getStats(self):
        """Returns any stats collected so far and resets them internally"""
        stats = self.stats
        self.stats = {}
        return stats


def _runOne(modelSpec, specRef, modelArg, constRefs, inputRefs, inline=False,
            completionQ=None, queryId=None, cacheModel=False, clientID=None,
            runPool=None, stats=None):
    """Issue one query asynchronously to ray, returns a future. inline will run
       all data processing steps in the same function as the model."""
    mClass = modelSpec.modelClass

    if inline:
        assert not modelSpec.modelType == 'kaas', "KaaS is not compatible with inline"
        assert runPool is None, "Cannot use run actors in inline mode"

        # We can't pass lists of references to ray functions because ray can't
        # statically determine the dataflow. All refs have to be first-class
        # arguments so we pack them all into a list and then expand it with
        # *varArgs
        if constRefs is None:
            varArgs = list(inputRefs)
        else:
            varArgs = list(constRefs) + list(inputRefs)

        if completionQ is not None:
            runInline.options(num_returns=mClass.nOutPost) \
                .remote(specRef, modelArg, *varArgs, completionQ=completionQ,
                        queryId=queryId)
            postOut = None
        else:
            postOut = runInline.options(num_returns=mClass.nOutPost) \
                .remote(specRef, modelArg, *varArgs)
    else:
        # Pre
        preInp = util.packInputs(mClass.preMap, const=constRefs, inp=inputRefs)

        preOut = pre.options(num_returns=mClass.nOutPre).remote(specRef, *preInp)
        if mClass.nOutPre == 1:
            preOut = [preOut]

        if PROF_LEVEL > 1:
            with infbench.timer("t_pre", stats):
                ray.wait(preOut, fetch_local=False)

        # Run
        runInp = util.packInputs(mClass.runMap, const=constRefs, inp=inputRefs, pre=preOut)
        # These are the only inputs that change from run to run, so we pass
        # them to the scheduler to determine readiness. We assume the consts
        # are always ready.
        dynInp = util.packInputs(mClass.runMap, inp=inputRefs, pre=preOut)

        if modelSpec.modelType == "kaas":
            model = modelArg
            with infbench.timer('t_kaas_generate_req', stats):
                req = ray.put(model.run(runInp, stats=stats))

            if completionQ is not None and mClass.noPost:
                runOut = runPool.run.options(num_returns=mClass.nOutRun). \
                    remote('runKaas', mClass.nOutRun, clientID, dynInp, [req],
                           {"queryId": queryId, "completionQ": completionQ, "clientID": clientID})
            else:
                runOut = runPool.run.options(num_returns=mClass.nOutRun). \
                    remote('runKaas', mClass.nOutRun, clientID, dynInp, [req],
                           {"clientID": clientID})
        else:  # Non-KaaS
            if completionQ is not None and mClass.noPost:
                runOut = runPool.run.options(num_returns=mClass.nOutRun). \
                    remote('runNative', mClass.nOutRun, clientID, dynInp,
                           [(specRef, modelArg)] + runInp,
                           {"completionQ": completionQ, "queryId": queryId,
                            "cacheModel": cacheModel, "clientID": clientID})
            else:
                runOut = runPool.run.options(num_returns=mClass.nOutRun). \
                    remote('runNative', mClass.nOutRun, clientID, dynInp,
                           [(specRef, modelArg), runInp], {"cacheModel": cacheModel})

        if mClass.nOutRun == 1:
            runOut = [runOut]

        if PROF_LEVEL > 1:
            with infbench.timer("t_run", stats):
                ray.wait(runOut, fetch_local=False)

        # Post
        if mClass.noPost:
            postOut = runOut
        else:
            postInp = util.packInputs(mClass.postMap, const=constRefs,
                                      inp=inputRefs, pre=preOut, run=runOut)
            postOut = post.options(num_returns=mClass.nOutPost) \
                .remote(specRef, *postInp, completionQ=completionQ, queryId=queryId)

            if mClass.nOutPost == 1:
                postOut = [postOut]

            if PROF_LEVEL > 1:
                with infbench.timer("t_post", stats):
                    ray.wait(postOut, fetch_local=False)

    return postOut


def _nShotAsync(n, loader, modelSpec, specRef, modelArg, constRefs, pool, benchConfig, stats):
    refs = []
    for i in range(n):
        idx = i % loader.ndata
        inp = loader.get(idx)

        # Ray is lazy and asynchronous so it's difficult to collect more
        # detailed metrics than e2e. Details within the remote functions
        # should match localBench results anyway.
        refs.append(_runOne(modelSpec, specRef, modelArg, constRefs, inp,
                    inline=benchConfig['inline'], runPool=pool,
                    cacheModel=benchConfig['cache'], stats=stats))

    # This isn't super accurate, but _runOne should return instantly and the
    # real work only happens when ray.get is called
    with infbench.timer('t_e2e', stats):
        results = []
        for i, ref in enumerate(refs):
            idx = i % loader.ndata

            # Dereference answer from post or the router's reference from run
            res = ray.get(ref)
            if modelSpec.modelClass.noPost:
                # Dereference the answer from run itself
                res = ray.get(res)

                if modelSpec.modelType == 'kaas':
                    res = maybeDereference(res)

            results.append((idx, res))

    return results


def _nShotSync(n, loader, modelSpec, specRef, modelArg, constRefs, pool, benchConfig, stats):
    results = []
    for i in range(n):
        idx = i % loader.ndata
        inp = loader.get(idx)

        with infbench.timer('t_e2e', stats):
            # Ray is lazy and asynchronous so it's difficult to collect more
            # detailed metrics than e2e. Details within the remote functions
            # should match localBench results anyway.
            res = _runOne(modelSpec, specRef, modelArg, constRefs, inp,
                          inline=benchConfig['inline'], runPool=pool,
                          cacheModel=benchConfig['cache'], stats=stats)

            # Dereference answer from post or the router's reference from run
            # (if nopost)
            res = ray.get(res)
            if modelSpec.modelClass.noPost:
                # Dereference the answer from run itself
                res = ray.get(res)

                if modelSpec.modelType == 'kaas':
                    res = maybeDereference(res)

        results.append((idx, res))

    return results


def nShot(modelSpec, n, benchConfig, reportPath="results.json"):
    ray.init(include_dashboard=False)

    coldStats = infbench.profCollection()
    warmStats = infbench.profCollection()

    specRef = ray.put(modelSpec)

    with infbench.timer("t_register_model", warmStats):
        constants = modelSpec.modelClass.getConstants(modelSpec.modelPath.parent)

        if constants is None:
            constRefs = None
        else:
            constRefs = []
            for const in constants:
                constRefs.append(ray.put(const))

        if modelSpec.modelType == "kaas":
            runConstRefs = []
            if modelSpec.modelClass.runMap.const is not None:
                for idx in modelSpec.modelClass.runMap.const:
                    runConstRefs.append(constRefs[idx])

            modelArg = modelSpec.getModelArg(constRefs=runConstRefs)
        else:
            modelArg = ray.put(modelSpec.getModelArg())

    with infbench.timer("t_init_loader", warmStats):
        loader = modelSpec.loader(modelSpec.dataDir)
        loader.preLoad(range(min(max(n, util.getNGpu()*2), loader.ndata)))

    nGpu = util.getNGpu()
    if USE_THREADED_POLICY:
        pool = policy.Pool.options(max_concurrency=policyNThread). \
            remote(nGpu, benchConfig['policy'], runActor)
    else:
        pool = policy.Pool.remote(nGpu, benchConfig['policy'], runActor)

    # Cold start metrics collection
    results = _nShotSync(1, loader, modelSpec, specRef, modelArg, constRefs, pool, benchConfig, coldStats)
    coldPoolStats = ray.get(pool.getStats.remote())
    coldStats.merge(coldPoolStats[None])

    # Make sure we're done with cold starts by running a large number of
    # requests. Done async to maximize the chances of everything getting warm
    # when there are multiple GPUs
    print(f"Running {2*util.getNGpu()} warmup passes")
    results = _nShotAsync(util.getNGpu()*2, loader, modelSpec, specRef,
                          modelArg, constRefs, pool, benchConfig, coldStats)
    # getting stats resets them for the warm runs
    ray.get(pool.getStats.remote())

    # Warm Runs
    print("Beginning warm runs")
    # results = _nShotAsync(n, loader, modelSpec, specRef, modelArg, constRefs, pool, benchConfig, warmStats)
    results = _nShotSync(n, loader, modelSpec, specRef, modelArg, constRefs, pool, benchConfig, warmStats)
    warmPoolStats = ray.get(pool.getStats.remote())

    if loader.checkAvailable:
        accuracies = [loader.check(res, idx) for idx, res in results]
        print("Accuracy = ", sum([int(res) for res in accuracies]) / n)
    else:
        print("Accuracy checking not supported by this dataset")

    warmStats.merge(warmPoolStats[None])

    print("\nDetailed Stats: ")
    print("Cold:")
    report = coldStats.report(includeEvents=False)
    util.analyzeStats(report)

    print("Warm:")
    report = warmStats.report(includeEvents=False)
    util.analyzeStats(report)

    if not isinstance(reportPath, pathlib.Path):
        reportPath = pathlib.Path(reportPath).resolve()

    print("Saving results to: ", reportPath)
    if reportPath.exists():
        reportPath.unlink()

    record = {
        "config": benchConfig,
        "metrics": report
    }

    with open(reportPath, 'w') as f:
        json.dump(record, f)

    # print("Ray Profiling:")
    # ray.timeline(filename="rayProfile.json")

    return results


# =============================================================================
# Throughput Test
# =============================================================================
class throughputLoop():
    def __init__(self, modelSpec, benchConfig, targetTime=60):
        """This test uses tornado IOLoop to submit requests as fast as possible
        for targetTime seconds. It reports the total throughput acheived."""
        self.benchConfig = benchConfig
        self.modelSpec = modelSpec
        self.clientID = benchConfig['name'].encode('utf-8')
        self.loop = IOLoop.instance()

        self.coldStats = infbench.profCollection()
        self.warmStats = infbench.profCollection()

        #
        # Common Inputs
        #
        constants = self.modelSpec.modelClass.getConstants(self.modelSpec.modelPath.parent)
        if constants is None:
            self.constRefs = None
        else:
            self.constRefs = []
            for const in constants:
                self.constRefs.append(ray.put(const))

        self.specRef = ray.put(self.modelSpec)

        if modelSpec.modelType == "kaas":
            runConstRefs = []
            if modelSpec.modelClass.runMap.const is not None:
                for idx in modelSpec.modelClass.runMap.const:
                    runConstRefs.append(self.constRefs[idx])

            self.modelArg = modelSpec.getModelArg(constRefs=runConstRefs)
        else:
            self.modelArg = ray.put(modelSpec.getModelArg())

        #
        # Common Test Components
        #
        self.loader = self.modelSpec.loader(self.modelSpec.dataDir)
        self.loader.preLoad(range(self.loader.ndata))

        self.nGpu = util.getNGpu()
        if USE_THREADED_POLICY:
            self.pool = policy.Pool.options(max_concurrency=policyNThread). \
                remote(self.nGpu, benchConfig['policy'], runActor)
        else:
            self.pool = policy.Pool.remote(self.nGpu, benchConfig['policy'], runActor)

        # This info is only used to get performance estimates
        gpuType = util.getGpuType()
        maxQps, _ = modelSpec.modelClass.getPerfEstimates(gpuType)

        self.completionQueue = ray.util.queue.Queue()

        #
        # Throughput Test Variables
        #
        # This can be a very rough estimate. It needs to be high enough that
        # the pipe stays full, but low enough that we aren't waiting for a
        # million queries to finish after the deadline.
        self.targetOutstanding = max(5, math.ceil(maxQps*benchConfig['scale']))
        self.targetTime = targetTime
        self.nOutstanding = 0
        self.nextIdx = 0
        self.nCompleted = 0

        #
        # Begin Test
        #
        self.preWarm()

        self.loop.add_callback(self.submitReqs)
        self.loop.add_callback(self.handleResponses)
        self.startTime = time.time()

    def preWarm(self):
        self.loader.preLoad([0])
        inputs = self.loader.get(0)
        results = []

        # We don't control how ray handles workers, but we assume that
        # sending a burst of nGpu*2 should be enough to trigger all the
        # cold starts.
        for i in range(self.nGpu*2):
            results.append(_runOne(self.modelSpec, self.specRef, self.modelArg,
                           self.constRefs, inputs, inline=self.benchConfig['inline'],
                           runPool=self.pool, stats=self.coldStats))
        for res in results:
            ray.get(res)

        coldPoolStats = ray.get(self.pool.getStats.remote())
        self.coldStats.merge(coldPoolStats[None])

    def submitReqs(self):
        while self.nOutstanding < self.targetOutstanding:
            inp = self.loader.get(self.nextIdx % self.loader.ndata)
            _runOne(self.modelSpec, self.specRef, self.modelArg,
                    self.constRefs, inp, inline=self.benchConfig['inline'],
                    completionQ=self.completionQueue, queryId=self.nextIdx,
                    cacheModel=self.benchConfig['cache'], runPool=self.pool,
                    stats=self.warmStats)

            self.nextIdx += 1
            self.nOutstanding += 1

    async def handleResponses(self):
        result, reqID = await self.completionQueue.get_async()

        if reqID % self.targetOutstanding == 0:
            if time.time() - self.startTime >= self.targetTime:
                print("Test complete, shutting down")
                self.runTime = time.time() - self.startTime
                IOLoop.instance().stop()

        self.nOutstanding -= 1
        self.nCompleted += 1

        if self.nOutstanding < (self.targetOutstanding / 2):
            self.loop.add_callback(self.submitReqs)

        self.loop.add_callback(self.handleResponses)

    def reportMetrics(self):
        metrics = {}

        # useful for debugging mostly. Ideally t_total ~= targetTime
        metrics['n_completed'] = self.nCompleted
        metrics['t_total'] = self.runTime * 1000  # we always report time in ms

        # completions/second
        metrics['throughput'] = self.nCompleted / self.runTime

        if self.nCompleted < self.targetOutstanding:
            print("\n*********************************************************")
            print("WARNING: Too few queries completed!")
            print("*********************************************************\n")
            metrics['valid'] = False
        elif self.runTime > self.targetTime*1.2 or self.runTime < self.targetTime*0.2:
            print("\n*********************************************************")
            print("WARNING: Actual runtime too far from target: ")
            print("\tTarget: ", self.targetTime)
            print("\tActual: ", self.runTime)
            print("*********************************************************\n")
            metrics['valid'] = False

        else:
            metrics['valid'] = True

        warmPoolStats = ray.get(self.pool.getStats.remote())
        self.warmStats.merge(warmPoolStats[None])

        return metrics


def throughput(modelSpec, benchConfig):
    ray.init(include_dashboard=False)

    if benchConfig['scale'] is None:
        benchConfig['scale'] = 1.0

    testLoop = throughputLoop(modelSpec, benchConfig, targetTime=20)
    IOLoop.instance().start()

    # XXX This isn't really a solution. We can check stats until the system
    # quiesces. I guess we should just wait till everything clears up before
    # reporting...
    time.sleep(2)
    metrics = testLoop.reportMetrics()

    report = testLoop.warmStats.report(includeEvents=False)
    util.analyzeStats(report)

    infbench.model.saveReport(metrics, benchConfig, benchConfig['name'] + '_results.json')


# =============================================================================
# MLPERF INFERENCE STUFF
# =============================================================================


def handleCompletion(modelSpec, queue):
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
        resp, qid = queue.get()

        if isinstance(resp, int):
            # The driver is asking us to wrap up and exit after we've seen
            # 'resps' many responses.
            targetComplete = resp

            if ncomplete == targetComplete:
                break
            else:
                wrapItUp = True
        else:
            # Normal completion message from a worker

            # Technically we should do this to load the result to the handler.
            # Unfortunately this causes an error in ray that I can't figure out
            # so we skip it.
            # if modelSpec.modelType == 'kaas' and modelSpec.modelClass.noPost:
            #     resp = handleKaasResult(resp)

            completion = mlperf_loadgen.QuerySampleResponse(qid, 0, 0)
            mlperf_loadgen.QuerySamplesComplete([completion])
            ncomplete += 1


class mlperfRunner():
    def __init__(self, modelSpec, loader, constantRefs, benchConfig):
        self.modelSpec = modelSpec
        self.loader = loader
        self.constants = constantRefs
        self.benchConfig = benchConfig

        self.coldStats = infbench.profCollection()
        self.warmStats = infbench.profCollection()

        if modelSpec.modelType == "kaas":
            runConstRefs = []
            if modelSpec.modelClass.runMap.const is not None:
                for idx in modelSpec.modelClass.runMap.const:
                    runConstRefs.append(self.constants[idx])

            self.modelArg = modelSpec.getModelArg(constRefs=runConstRefs)
        else:
            self.modelArg = ray.put(modelSpec.getModelArg())

        # Total number of queries issued
        self.nIssued = 0

        self.specRef = ray.put(self.modelSpec)

        self.nGpu = util.getNGpu()

        if USE_THREADED_POLICY:
            self.pool = policy.Pool.options(max_concurrency=policyNThread). \
                remote(self.nGpu, benchConfig['policy'], runActor)
        else:
            self.pool = policy.Pool.remote(self.nGpu, benchConfig['policy'], runActor)

    def start(self, preWarm=True):
        self.completionQueue = ray.util.queue.Queue()
        self.completionHandler = threading.Thread(
                target=handleCompletion, args=[self.modelSpec, self.completionQueue])
        self.completionHandler.start()

        # This is very important for Ray because the cold start is multiple
        # seconds and mlperf is based on SLOs which we violate immediately.
        if preWarm:
            self.loader.preLoad([0])
            inputs = self.loader.get(0)
            results = []

            # We don't control how ray handles workers, but we assume that
            # sending a burst of nGpu*2 should be enough to trigger all the
            # cold starts.
            for i in range(self.nGpu*2):
                results.append(_runOne(self.modelSpec, self.specRef, self.modelArg,
                               self.constants, inputs, inline=self.benchConfig['inline'],
                               runPool=self.pool, stats=self.coldStats))
            for res in results:
                ray.get(res)

            coldPoolStats = ray.get(self.pool.getStats.remote())
            self.coldStats.merge(coldPoolStats[None])

    def runBatch(self, queryBatch):
        for q in queryBatch:
            inp = self.loader.get(q.index)

            _runOne(self.modelSpec, self.specRef, self.modelArg,
                    self.constants, inp, inline=self.benchConfig['inline'],
                    completionQ=self.completionQueue, queryId=q.id,
                    cacheModel=self.benchConfig['cache'], runPool=self.pool, stats=self.warmStats)

        self.nIssued += len(queryBatch)

    def processLatencies(self, latencies):
        self.latMetrics = infbench.model.processLatencies(self.benchConfig, latencies)

    def stop(self):
        self.completionQueue.put((self.nIssued, None))
        print("Waiting for completion handler to finish")
        self.completionHandler.join()

        warmPoolStats = ray.get(self.pool.getStats.remote())
        self.warmStats.merge(warmPoolStats[None])

        return (self.coldStats, self.warmStats)


def mlperfBench(modelSpec, benchConfig):
    """Run the mlperf loadgen version"""
    ray.init(include_dashboard=False)

    gpuType = util.getGpuType()
    settings = modelSpec.modelClass.getMlPerfCfg(gpuType, benchConfig)
    loader = modelSpec.loader(modelSpec.dataDir)

    constants = modelSpec.modelClass.getConstants(modelSpec.modelPath.parent)
    if constants is None:
        constRefs = None
    else:
        constRefs = []
        for const in constants:
            constRefs.append(ray.put(const))

    runner = mlperfRunner(modelSpec, loader, constRefs, benchConfig)

    runner.start(preWarm=True)
    sut = mlperf_loadgen.ConstructSUT(
        runner.runBatch, infbench.model.flushQueries, runner.processLatencies)

    qsl = mlperf_loadgen.ConstructQSL(
        loader.ndata, infbench.model.mlperfNquery, loader.preLoad, loader.unLoad)

    mlperf_loadgen.StartTest(sut, qsl, settings)
    mlperf_loadgen.DestroyQSL(qsl)
    mlperf_loadgen.DestroySUT(sut)

    coldStats, warmStats = runner.stop()

    print("\nResults:")
    mlPerfMetrics = infbench.model.parseMlPerf('mlperf_log_')

    print("\nStats:")
    report = warmStats.report()
    util.analyzeStats(report)

    infbench.model.saveReport({**runner.latMetrics, **mlPerfMetrics}, benchConfig, 'results.json')


# =============================================================================
# Server Mode
# =============================================================================


class clientState():
    def __init__(self, modelName):
        self.modelSpec = util.getModelSpec(modelName)
        self.specRef = ray.put(self.modelSpec)

        constants = self.modelSpec.modelClass.getConstants(self.modelSpec.modelPath.parent)
        if constants is None:
            constRefs = None
        else:
            constRefs = []
            for const in constants:
                constRefs.append(ray.put(const))
        self.constRefs = constRefs

        if self.modelSpec.modelType == "kaas":
            runConstRefs = []
            if self.modelSpec.modelClass.runMap.const is not None:
                for idx in self.modelSpec.modelClass.runMap.const:
                    runConstRefs.append(self.constRefs[idx])

            self.modelArg = self.modelSpec.getModelArg(constRefs=runConstRefs)
        else:
            self.modelArg = ray.put(self.modelSpec.getModelArg())


# { clientID -> clientState }
# For now, we only support one model per client
clients = {}


@ray.remote(num_gpus=1)
def fakeRayModel(clientID, reqID, rayQ):
    time.sleep(.07)
    rayQ.put(([b''], (clientID, reqID)))


@ray.remote
class fakeActorModel():
    def run(self, clientID, reqID, rayQ):
        time.sleep(.07)
        rayQ.put(([b''], (clientID, reqID)))

        # Return value is used as a completion notification
        return True


class serverLoop():
    """ServerTask"""
    def __init__(self, clientSock, barrierSock, benchConfig):
        self.loop = IOLoop.instance()
        self.benchConfig = benchConfig

        self.clientStream = ZMQStream(clientSock)
        self.clientStream.on_recv(self.handleClients)

        self.barrierStream = ZMQStream(barrierSock)
        self.barrierStream.on_recv(self.handleBarrier)
        self.readyClients = []

        IOLoop.current().add_callback(self.handleWorker)

        self.nGpu = util.getNGpu()

        self.clientStats = {}

        if USE_THREADED_POLICY:
            self.pool = policy.Pool.options(max_concurrency=policyNThread). \
                remote(self.nGpu, benchConfig['policy'], runActor)
        else:
            self.pool = policy.Pool.remote(self.nGpu, benchConfig['policy'], runActor)

        self.rayQ = ray.util.queue.Queue()

        self.nOutstanding = 0
        self.overwhelmed = False

    def handleBarrier(self, msg):
        clientID = msg[0]

        print("Recieved Ready from: ", clientID.decode("utf-8"))
        self.readyClients.append(clientID)
        if len(self.readyClients) == self.benchConfig['numClient']:
            # Get cold-start stats (if any) and reset for main warm passes
            poolStats = ray.get(self.pool.getStats.remote())
            self.coldStats = self.clientStats
            util.mergePerClientStats(self.coldStats, poolStats)
            self.clientStats = {}

            print("Releasing Barrier")
            for cID in self.readyClients:
                self.barrierStream.send_multipart([cID, b'', b'GO'])

    async def handleWorker(self):
        result, reqData = await self.rayQ.get_async()
        clientID = reqData[0]
        reqID = reqData[1]

        # Ideally, ray would handle this in their Queue implementation but they
        # can't recurse into datastructures so we have to fetch the result
        # here. It's guaranteed to be ready (since any references come from
        # KaaS which already put them into the kv store) but we do have to wait
        # for the data transfer.
        result = maybeDereference(result)

        outBufs = [clientID, reqID]
        outBufs.extend(result)
        self.clientStream.send_multipart(outBufs)
        IOLoop.current().add_callback(self.handleWorker)

        self.nOutstanding -= 1
        # Start accepting work again once we get below our max (plus some
        # hysteresis)
        if self.overwhelmed and self.nOutstanding < (maxOutstanding * 0.8):
            self.clientStream.on_recv(self.handleClients)

    async def fakeModel(self, clientID, reqID, startTime=None):
        startTime = time.time()
        await self.sem.acquire()
        self.profs['t_queue'].increment(time.time() - startTime)

        await asyncio.sleep(0.070)
        self.sem.release()

        self.profs['t_e2e'].increment(time.time() - startTime)
        self.clientStream.send_multipart([clientID, reqID, b''])

    def handleClients(self, msg):
        clientID = msg[0]
        reqID = msg[1]
        data = msg[2:]
        # clientID, reqID, data = msg

        cState = clients.get(clientID, None)

        if clientID not in self.clientStats:
            self.clientStats[clientID] = infbench.profCollection()

        if cState is None:
            # Registration
            print("Registering ", clientID)
            modelName = reqID.decode('utf-8')
            cState = clientState(modelName)
            clients[clientID] = cState
        else:
            self.nOutstanding += 1
            # Too many outstanding queries can overwhelm Ray and hurt
            # throughput.
            if self.nOutstanding > maxOutstanding:
                self.clientStream.stop_on_recv()
                self.overwhelmed = True

            _runOne(cState.modelSpec, cState.specRef, cState.modelArg,
                    cState.constRefs, data, completionQ=self.rayQ,
                    queryId=(clientID, reqID), clientID=clientID,
                    cacheModel=self.benchConfig['cache'],
                    inline=self.benchConfig['inline'], runPool=self.pool,
                    stats=self.clientStats[clientID])

    def shutdown(self):
        self.clientStream.stop_on_recv()
        IOLoop.instance().stop()

        poolStats = ray.get(self.pool.getStats.remote())
        self.warmStats = self.clientStats
        util.mergePerClientStats(self.warmStats, poolStats)
        self.clientStats = {}


def serveRequests(benchConfig):
    ray.init(include_dashboard=False)
    context = zmq.Context()

    clientSock = context.socket(zmq.ROUTER)
    clientSock.bind(util.clientUrl)

    barrierSock = context.socket(zmq.ROUTER)
    barrierSock.bind(util.barrierUrl)

    # IOLoop uses a global context, when you instantiate a serverLoop object,
    # it registers itself with IOLoop. The returned object isn't used here.
    looper = serverLoop(clientSock, barrierSock, benchConfig)

    signal.signal(signal.SIGINT, lambda s, f: IOLoop.instance().add_callback_from_signal(looper.shutdown))

    print("Beginning serving loop")
    IOLoop.instance().start()
    print("Server Exiting")

    # print("Stats:")
    # for cID, stats in looper.warmStats.items():
    #     print("Client: ", cID)
    #     util.analyzeStats(stats.report(includeEvents=False))
