import ray
import ray.util.queue
import threading
import os
import sys
import signal
import math
from pprint import pprint
import psutil
import enum
import collections

import time

from tornado.ioloop import IOLoop
import zmq
from zmq.eventloop.zmqstream import ZMQStream

import mlperf_loadgen
import kaas
import kaas.ray
import kaas.pool
from kaas import profiling

import infbench
from infbench import properties

import util


# Prof levels control the level of detail recorded, higher levels may have an
# impact on performance.
PROF_LEVEL = 1  # minimal performance impact
# PROF_LEVEL = 2  # serializes a lot of stuff, really slows down e2e

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


# Maximum memory (bytes) and SM (percent) that can be allocated from one GPU
maxMemResource = 16160 * (2**20)
maxSMResource = 100


def roundMIG(frac, ceil=True):
    # Round up to the nearest multiple of 1/8 (granularity of MIG slices)
    if frac % 0.125 == 0:
        nSlice = frac / 0.125
    else:
        if ceil:
            nSlice = int(frac / 0.125) + 1
        else:
            nSlice = int(frac / 0.125)

    return nSlice * 0.125


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


# Recursively fetch all ray references in refs. refs may be a regular value,
# Ray reference, or list or tuple of regular values and/or Ray references.
def flattenRayRefs(refs):
    if isinstance(refs, ray._raylet.ObjectRef):
        return flattenRayRefs(ray.get(refs))
    elif isinstance(refs, list):
        return [flattenRayRefs(ref) for ref in refs]
    elif isinstance(refs, tuple):
        return tuple([flattenRayRefs(ref) for ref in refs])
    else:
        return refs


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


@kaas.pool.remote_with_confirmation()
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


def _run(model, inputs, completionQ, queryId, profs=None):
    """Internal run function"""
    constants, data = _unMarshalArgs(model.runMap, inputs)

    with profiling.timer('t_model_run', profs):
        results = model.run(constants + list(data), stats=profs)

    if completionQ is not None:
        completionQ.put((results, queryId))

    return results


@ray.remote(num_gpus=1)
def runKaasTask(req, queryId=None, completionQ=None):
    results = kaas.ray.invoke(req.toDict())

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


@kaas.pool.remote_with_confirmation()
def post(modelSpec, *inputs, completionQ=None, queryId=None):
    mClass = modelSpec.modelClass
    constants, rawData = _unMarshalArgs(mClass.postMap, inputs)

    data = flattenRayRefs(rawData)

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
class runActor(kaas.pool.PoolWorker):
    """A persistent actor for running model requests. Actors will cache models
    as needed and run them natively. It is possible to run out of GPU memory
    with actors since they cache every model they are passed."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.modelCache = {}
        self.kaasReady = False

    def runNative(self, modelInfo, inputRefs, completionQ=None, queryId=None,
                  cacheModel=False, clientID=None):

        # For MPS modes, CUDA_VISIBLE_DEVICES cannot be set so we use an
        # explicit devID instead. It doesn't hurt to do it this way for all
        # modes though.
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            self.devID = int(os.environ['CUDA_VISIBLE_DEVICES'])
            del os.environ['CUDA_VISIBLE_DEVICES']

        profs = self.getProfs()

        # The runActor must cache the model, if you wan't to reset, you must
        # kill and restart the actor. cacheModel is kept for consistency with
        # runTask but is ignored here.
        if clientID in self.modelCache:
            model, consts = self.modelCache[clientID]

            if model.runMap.const is None:
                nConst = 0
            else:
                nConst = len(model.runMap.const)
        else:
            with profiling.timer("t_model_init", profs):
                with profiling.timer('t_loadInput', profs, final=False):
                    modelSpec = ray.get(modelInfo[0])
                    modelArg = ray.get(modelInfo[1])

                model = modelSpec.modelClass(modelArg, devID=self.devID)

            if modelSpec.modelClass.runMap.const is None:
                nConst = 0
            else:
                nConst = len(modelSpec.modelClass.runMap.const)

            with profiling.timer('t_loadInput', profs, final=False):
                consts = ray.get(inputRefs[:nConst])

            self.modelCache[clientID] = (model, consts)

        with profiling.timer('t_loadInput', profs):
            inputs = ray.get(inputRefs[nConst:])

        result = _run(model, consts + inputs, completionQ, queryId, profs=profs)
        assert isinstance(result, tuple)

        with profiling.timer('t_writeOutput', profs):
            resRefs = [ray.put(res) for res in result]

        return tuple(resRefs)

    def runInline(self, modelInfo, constRefs, inputRefs, completionQ=None, queryId=None,
                  cacheModel=False, clientID=None):
        profs = self.getProfs()

        # The runActor must cache the model, if you wan't to reset, you must
        # kill and restart the actor. cacheModel is kept for consistency with
        # runTask but is ignored here.
        if clientID in self.modelCache:
            model = self.modelCache[clientID]
        else:
            with profiling.timer("t_model_init", profs):
                with profiling.timer('t_loadInput', profs, final=False):
                    modelSpec = ray.get(modelInfo[0])
                    modelArg = ray.get(modelInfo[1])
                model = modelSpec.modelClass(modelArg)
                self.modelCache[clientID] = model

        with profiling.timer('t_loadInput', profs):
            inputs = ray.get(inputRefs)

        inputs = ray.get(inputRefs)
        consts = ray.get(constRefs)

        preInp = util.packInputs(model.preMap, const=consts, inp=inputs)
        preOut = model.pre(preInp)

        runInp = util.packInputs(model.runMap, const=consts, inp=inputs, pre=preOut)
        runOut = model.run(runInp)

        if model.noPost:
            result = runOut
        else:
            postInp = util.packInputs(model.postMap, const=constRefs,
                                      inp=inputRefs, pre=preOut, run=runOut)

            result = model.post(postInp)

        if completionQ is not None:
            completionQ.put((result, queryId))

        return tuple(result)

    def runKaas(self, req, queryId=None, completionQ=None, clientID=None):
        profs = self.getProfs()

        if not self.kaasReady:
            kaas.ray.init()
            self.kaasReady = True

        with profiling.timer('t_model_run', profs):
            results = kaas.ray.invoke(req, profs=profs.mod('kaas'), clientID=clientID)

        if completionQ is not None:
            completionQ.put((results, queryId))

        return results

    def shutdown(self):
        for modelInfo in self.modelCache.values():
            modelInfo[0].shutdown()


def _runOne(modelSpec, specRef, modelArg, constRefs, inputRefs, inline=False,
            completionQ=None, queryId=None, cacheModel=False, clientID=None,
            runPool=None, stats=None):
    """Issue one query asynchronously to ray, returns a future. inline will run
       all data processing steps in the same function as the model."""
    mClass = modelSpec.modelClass

    if inline:
        assert not modelSpec.modelType == 'kaas', "KaaS is not compatible with inline"

        if mClass.noPost:
            nOut = mClass.nOutRun
        else:
            nOut = mClass.nOutPost

        postOut = runPool.run(clientID, 'runInline', num_returns=mClass.nOutRun,
                              refDeps=inputRefs,
                              args=[(specRef, modelArg), constRefs, inputRefs],
                              kwargs={"completionQ": completionQ, "queryId": queryId,
                                      "cacheModel": cacheModel, "clientID": clientID})

        if nOut == 1:
            postOut = [postOut]

        if PROF_LEVEL > 1:
            with profiling.timer("t_run", stats):
                ray.wait(postOut, fetch_local=False)
    else:
        # Pre
        if mClass.noPre:
            preOut = inputRefs
            dynInp = []
            if PROF_LEVEL > 1 and stats is not None:
                stats["t_pre"].increment(0)

        else:
            preInp = util.packInputs(mClass.preMap, const=constRefs, inp=inputRefs)

            preOut = pre.options(num_returns=mClass.nOutPre + 1).remote(specRef, *preInp)
            preConfirm = preOut[0]
            preOut = preOut[1:]

            dynInp = [preConfirm]

            if PROF_LEVEL > 1:
                with profiling.timer("t_pre", stats):
                    ray.wait(dynInp, fetch_local=False)

        # Run
        runInp = util.packInputs(mClass.runMap, const=constRefs, inp=inputRefs, pre=preOut)

        if modelSpec.modelType == "kaas":
            model = modelArg
            with profiling.timer('t_kaas_generate_req', stats):
                reqRef = ray.put(model.run(runInp, stats=stats))

            if completionQ is not None and mClass.noPost:
                runOut = runPool.run(clientID, 'runKaas', num_returns=mClass.nOutRun,
                                     refDeps=dynInp,
                                     args=[reqRef],
                                     kwargs={"queryId": queryId,
                                             "completionQ": completionQ,
                                             "clientID": clientID})
            else:
                runOut = runPool.run(clientID, 'runKaas', num_returns=mClass.nOutRun,
                                     refDeps=dynInp,
                                     args=[reqRef],
                                     kwargs={"clientID": clientID})
        else:  # Non-KaaS
            if completionQ is not None and mClass.noPost:
                runOut = runPool.run(clientID, 'runNative', num_returns=mClass.nOutRun,
                                     refDeps=dynInp,
                                     args=[(specRef, modelArg), runInp],
                                     kwargs={"completionQ": completionQ,
                                             "queryId": queryId,
                                             "cacheModel": cacheModel,
                                             "clientID": clientID})
            else:
                runOut = runPool.run(clientID, 'runNative', num_returns=mClass.nOutRun,
                                     refDeps=dynInp,
                                     args=[(specRef, modelArg), runInp],
                                     kwargs={"cacheModel": cacheModel})

        if mClass.nOutRun == 1:
            runOut = [runOut]

        if PROF_LEVEL > 1:
            with profiling.timer("t_run", stats):
                resRefs = ray.get(runOut)
                ray.wait(list(resRefs), fetch_local=False)

        # Post
        if mClass.noPost:
            postOut = runOut
            if PROF_LEVEL > 1 and stats is not None:
                stats["t_post"].increment(0)
        else:
            postInp = util.packInputs(mClass.postMap, const=constRefs,
                                      inp=inputRefs, pre=preOut, run=runOut)
            allOut = post.options(num_returns=mClass.nOutPost + 1) \
                .remote(specRef, *postInp, completionQ=completionQ, queryId=queryId)

            postConfirm = allOut[0]
            postOut = allOut[1:]

            if PROF_LEVEL > 1:
                with profiling.timer("t_post", stats):
                    ray.wait([postConfirm], fetch_local=False)

    return postOut


def warmKaas(pool):
    modelSpec = util.getModelSpec("dummyModel", "kaas")
    specRef = ray.put(modelSpec)
    modelArg = modelSpec.getModelArg()
    loader = modelSpec.loader(modelSpec.dataDir)
    loader.preLoad(0)

    inpRefs = [ray.put(val) for val in loader.get(0)]
    resRefs = _runOne(modelSpec, specRef, modelArg, [], inpRefs, runPool=pool)
    res = flattenRayRefs(resRefs)

    # clear stats after dummy
    pool.getProfile()

    assert loader.check(res, 0)


def _nShotAsync(n, loader, modelSpec, specRef, modelArg, constRefs, pool, benchConfig, stats):
    refs = []
    cachedInputs = {}
    for i in range(n):
        idx = i % loader.ndata
        if modelSpec.cacheInputs:
            if idx not in cachedInputs:
                inputs = loader.get(idx)
                cachedInputs[i] = [ray.put(val) for val in inputs]
            inpRefs = cachedInputs[idx]
        else:
            inputs = loader.get(idx)
            inpRefs = [ray.put(val) for val in inputs]

        # Ray is lazy and asynchronous so it's difficult to collect more
        # detailed metrics than e2e. Details within the remote functions
        # should match localBench results anyway.
        refs.append(_runOne(modelSpec, specRef, modelArg, constRefs, inpRefs,
                    inline=benchConfig['inline'], runPool=pool, clientID=benchConfig['name'],
                    cacheModel=benchConfig['forceCold'], stats=stats))

    # This isn't super accurate, but _runOne should return instantly and the
    # real work only happens when ray.get is called
    with profiling.timer('t_e2e', stats):
        results = []
        for i, ref in enumerate(refs):
            idx = i % loader.ndata

            # Dereference answer from post or the router's reference from run
            res = flattenRayRefs(ref)

            results.append((idx, res))

    return results


def _nShotSync(n, loader, modelSpec, specRef, modelArg, constRefs, pool, benchConfig, stats):
    results = []
    cachedInputs = {}
    for i in range(n):
        idx = i % loader.ndata
        if modelSpec.cacheInputs:
            if idx not in cachedInputs:
                inputs = loader.get(idx)
                cachedInputs[i] = [ray.put(val) for val in inputs]
            inpRefs = cachedInputs[idx]
        else:
            inputs = loader.get(idx)
            inpRefs = [ray.put(val) for val in inputs]

        with profiling.timer('t_e2e', stats):
            # Ray is lazy and asynchronous so it's difficult to collect more
            # detailed metrics than e2e. Details within the remote functions
            # should match localBench results anyway.
            resRefs = _runOne(modelSpec, specRef, modelArg, constRefs, inpRefs,
                              clientID=benchConfig['name'], inline=benchConfig['inline'], runPool=pool,
                              cacheModel=benchConfig['forceCold'], stats=stats)

            res = flattenRayRefs(resRefs)

        results.append((idx, res))

    return results


def nShotRegister(pool, modelSpec, benchConfig):
    if benchConfig['fractional'] is None:
        pool.registerGroup(benchConfig['name'], runActor)
    else:
        nGPU = infbench.getNGpu()
        props = properties.getProperties()
        reqs = props.resourceReqs(modelSpec.name, modelSpec.modelType)

        memFrac = roundMIG(reqs['mem'] / maxMemResource)
        smFrac = roundMIG(reqs['sm'] / maxSMResource)

        if memFrac > nGPU:
            raise RuntimeError("Maximum GPU memory exceeded")

        if benchConfig['fractional'] == 'mem':
            frac = memFrac
        else:
            frac = smFrac

        nWorkers = nGPU / frac
        if nWorkers == 0:
            raise RuntimeError(f"Insufficient resources for this client:\n\trequested: {frac}\n\tMaximum Allocation: {nGPU}")

        # We need at least one CPU for the pool and one more to execute tasks
        maxWorkers = psutil.cpu_count() - 2
        if nWorkers > maxWorkers:
            print(f"Warning: Maximum number of CPUs exceeded, reducing allocation from {nWorkers} to {maxWorkers}")
            nWorkers = maxWorkers

        pool.registerGroup(benchConfig['name'], runActor.options(num_gpus=frac),
                           nWorker=nWorkers, workerResources=frac)


def nShot(modelSpec, n, benchConfig, reportPath="results.json"):
    ray.init(include_dashboard=False)

    coldStats = profiling.profCollection()
    warmStats = profiling.profCollection()

    specRef = ray.put(modelSpec)

    with profiling.timer("t_register_model", warmStats):
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

    with profiling.timer("t_init_loader", warmStats):
        loader = modelSpec.loader(modelSpec.dataDir)
        loader.preLoad(range(min(max(n, infbench.getNGpu()*2), loader.ndata)))

    nGPU = infbench.getNGpu()
    pool = kaas.pool.Pool(nGPU, policy=benchConfig['policy'])

    nShotRegister(pool, modelSpec, benchConfig)
    if modelSpec.modelType == 'kaas':
        warmKaas(pool)

    # Cold start metrics collection
    results = _nShotSync(1, loader, modelSpec, specRef, modelArg, constRefs, pool, benchConfig, coldStats)
    coldPoolStats = pool.getProfile()
    finalCold = profiling.profCollection()
    finalCold.mod('server').merge(coldStats)
    finalCold.merge(coldPoolStats)

    # Make sure we're done with cold starts by running a large number of
    # requests. Done async to maximize the chances of everything getting warm
    # when there are multiple GPUs
    print(f"Running {2*infbench.getNGpu()} warmup passes")
    results = _nShotAsync(infbench.getNGpu()*2, loader, modelSpec, specRef,
                          modelArg, constRefs, pool, benchConfig, None)

    # getting stats resets them for the warm runs
    pool.getProfile()

    # Warm Runs
    print("Beginning warm runs")
    # results = _nShotAsync(n, loader, modelSpec, specRef, modelArg, constRefs, pool, benchConfig, warmStats)
    results = _nShotSync(n, loader, modelSpec, specRef, modelArg, constRefs, pool, benchConfig, warmStats)
    warmPoolStats = pool.getProfile()

    finalWarm = profiling.profCollection()
    finalWarm.mod('server').merge(warmStats)
    finalWarm.merge(warmPoolStats)

    coldReport = finalCold.report(includeEvents=False, metrics=['mean'])
    warmReport = finalWarm.report(includeEvents=False, metrics=['mean'])

    print("Cold Results: ")
    pprint(coldReport)

    print("Warm Results: ")
    pprint(warmReport)

    print("Saving results to ", reportPath)
    infbench.saveReport(finalWarm, finalCold, benchConfig, reportPath)

    if loader.checkAvailable:
        accuracies = [loader.check(res, idx) for idx, res in results]
        print("Accuracy = ", sum([int(res) for res in accuracies]) / n)
    else:
        print("Dataset does not support accuracy calculation")

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

        self.initStats()
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

        if self.modelSpec.cacheInputs:
            self.inputRefs = {}

        #
        # Common Test Components
        #
        self.loader = self.modelSpec.loader(self.modelSpec.dataDir)
        self.loader.preLoad(range(self.loader.ndata))

        self.nGPU = infbench.getNGpu()
        self.pool = kaas.pool.Pool(self.nGPU, policy=benchConfig['policy'])
        self.pool.registerGroup("throughputGroup", runActor)

        # This info is only used to get performance estimates
        maxQps = properties.getProperties().throughputSingle(modelSpec.name, benchConfig['expKey'])

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

    def initStats(self):
        self.coldStats = profiling.profCollection()
        self.coldServerStats = self.coldStats.mod('server')
        self.warmStats = profiling.profCollection()
        self.warmServerStats = self.warmStats.mod('server')

    def preWarm(self):
        self.loader.preLoad([0])
        inputs = self.loader.get(0)
        inpRefs = [ray.put(val) for val in inputs]

        results = []

        # We don't control how ray handles workers, but we assume that
        # sending a burst of nGPU*2 should be enough to trigger all the
        # cold starts.
        for i in range(self.nGPU*2):
            results.append(_runOne(self.modelSpec, self.specRef, self.modelArg,
                           self.constRefs, inpRefs, inline=self.benchConfig['inline'],
                           runPool=self.pool, stats=self.coldServerStats))
        for res in results:
            ray.get(res)

        coldPoolStats = self.pool.getProfile()
        self.coldServerStats.merge(coldPoolStats.mod("throughputGroup"))

    def submitReqs(self):
        while self.nOutstanding < self.targetOutstanding:
            idx = self.nextIdx % self.loader.ndata
            if self.modelSpec.cacheInputs:
                if idx not in self.inputRefs:
                    inputs = self.loader.get(self.nextIdx % self.loader.ndata)
                    inpRefs = [ray.put(val) for val in inputs]
                    self.inputRefs[idx] = inpRefs
                inpRefs = self.inputRefs[idx]
            else:
                inputs = self.loader.get(self.nextIdx % self.loader.ndata)
                inpRefs = [ray.put(val) for val in inputs]

            _runOne(self.modelSpec, self.specRef, self.modelArg,
                    self.constRefs, inpRefs, inline=self.benchConfig['inline'],
                    completionQ=self.completionQueue, queryId=self.nextIdx,
                    cacheModel=self.benchConfig['forceCold'], runPool=self.pool,
                    stats=self.warmServerStats, clientID='throughputGroup')

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
        # metrics = profiling.profCollection()

        # useful for debugging mostly. Ideally t_total ~= targetTime
        self.warmServerStats['n_completed'].increment(self.nCompleted)
        self.warmServerStats['t_total'].increment(self.runTime * 1000)  # we always report time in ms

        # completions/second
        self.warmServerStats['throughput'].increment(self.nCompleted / self.runTime)

        if self.nCompleted < self.targetOutstanding:
            print("\n*********************************************************")
            print("WARNING: Too few queries completed!")
            print("*********************************************************\n")
            valid = False
        elif self.runTime > self.targetTime*1.2 or self.runTime < self.targetTime*0.2:
            print("\n*********************************************************")
            print("WARNING: Actual runtime too far from target: ")
            print("\tTarget: ", self.targetTime)
            print("\tActual: ", self.runTime)
            print("*********************************************************\n")
            valid = False

        else:
            valid = True

        warmPoolStats = self.pool.getProfile()
        self.warmStats.merge(warmPoolStats)

        currentStats = self.warmStats
        self.initStats()

        return currentStats, valid


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

    profs, valid = testLoop.reportMetrics()

    print("Saving results to ", benchConfig['name'] + '_results.json')
    infbench.saveReport(profs, None, benchConfig, benchConfig['name'] + '_results.json')

    print("Results")
    pprint(profs.report())


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

        self.coldStats = profiling.profCollection(detail=True)
        self.warmStats = profiling.profCollection(detail=True)
        self.serverColdStats = self.coldStats.mod('server')
        self.serverWarmStats = self.warmStats.mod('server')

        if self.modelSpec.cacheInputs:
            self.inputRefs = {}

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

        self.nGPU = infbench.getNGpu()
        self.pool = kaas.pool.Pool(self.nGPU, policy=benchConfig['policy'], profLevel=1)
        self.pool.registerGroup(benchConfig['name'], runActor)

    def start(self, preWarm=True):
        self.completionQueue = ray.util.queue.Queue()
        self.completionHandler = threading.Thread(target=handleCompletion,
                                                  args=[self.modelSpec, self.completionQueue])
        self.completionHandler.start()

        # This is very important for Ray because the cold start is multiple
        # seconds and mlperf is based on SLOs which we violate immediately.
        if preWarm:
            self.loader.preLoad([0])
            inputs = self.loader.get(0)
            inpRefs = [ray.put(val) for val in inputs]
            results = []

            # We don't control how ray handles workers, but we assume that
            # sending a burst of nGPU*2 should be enough to trigger all the
            # cold starts.
            for i in range(self.nGPU*2):
                results.append(_runOne(self.modelSpec, self.specRef, self.modelArg,
                               self.constants, inpRefs, inline=self.benchConfig['inline'],
                               runPool=self.pool,
                               stats=self.serverColdStats, clientID=self.benchConfig['name']))
            for res in results:
                ray.get(res)

            coldPoolStats = self.pool.getProfile()
            self.coldStats.merge(coldPoolStats)

    def runBatch(self, queryBatch):
        for q in queryBatch:
            if self.modelSpec.cacheInputs:
                if q.index in self.inputRefs:
                    inpRefs = self.inputRefs[q.index]
                else:
                    inputs = self.loader.get(q.index)
                    inpRefs = [ray.put(val) for val in inputs]
                    self.inputRefs[q.index] = inpRefs
            else:
                inputs = self.loader.get(q.index)
                inpRefs = [ray.put(val) for val in inputs]

            _runOne(self.modelSpec, self.specRef, self.modelArg,
                    self.constants, inpRefs, inline=self.benchConfig['inline'],
                    completionQ=self.completionQueue, queryId=q.id,
                    cacheModel=self.benchConfig['forceCold'], runPool=self.pool,
                    stats=self.serverWarmStats, clientID=self.benchConfig['name'])

        self.nIssued += len(queryBatch)

    def processLatencies(self, latencies):
        self.latMetrics = infbench.processLatencies(self.benchConfig, latencies)

    def stop(self):
        self.completionQueue.put((self.nIssued, None))
        print("Waiting for completion handler to finish")
        self.completionHandler.join()

        warmPoolStats = self.pool.getProfile()
        self.warmStats.merge(warmPoolStats)

        return (self.coldStats, self.warmStats)


def mlperfBench(modelSpec, benchConfig):
    """Run the mlperf loadgen version"""
    ray.init(include_dashboard=False)

    props = properties.getProperties()
    settings = props.getMlPerfConfig(modelSpec.name, benchConfig)
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
    mlPerfMetrics, valid = infbench.parseMlPerf('mlperf_log_')
    benchConfig['valid'] = valid

    if not valid:
        print("\n*********************************************************")
        print("WARNING: Results invalid, reduce target QPS and try again")
        print("*********************************************************\n")

    print("\nStats:")
    warmStats.mod('server')['t_response'] = runner.latMetrics
    warmStats.mod('server').merge(mlPerfMetrics)

    print("Performance Profilers")
    pprint(warmStats.report(metrics=['p50']))

    print("MLPerf Results:")
    pprint(mlPerfMetrics.report())

    infbench.saveReport(warmStats, None, benchConfig, 'results.json')


# =============================================================================
# Server Mode
# =============================================================================
class ServerError(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


class clientState():
    def __init__(self, clientID, modelName, modelType, fractional=None):
        self.clientID = clientID
        self.modelName = modelName
        self.modelSpec = util.getModelSpec(modelName, modelType)
        self.specRef = ray.put(self.modelSpec)

        # This is a work-around for Ray's immutable object store that can fill
        # up when models have even modestly sized inputs.
        if self.modelSpec.cacheInputs:
            self.loader = self.modelSpec.loader(self.modelSpec.dataDir)
            self.loader.preLoad(range(self.loader.ndata))
            self.cachedInputs = {}

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

        if fractional is not None:
            self.nWorker = 0
            self.memFrac, self.smFrac = self.getGPUFraction()
            # resFrac is the fraction of the target resource for this run
            if fractional == 'full':
                self.resFrac = 1.0
            elif fractional == 'mem':
                self.resFrac = roundMIG(self.memFrac)
            else:
                self.resFrac = roundMIG(self.smFrac)

    def getGPUFraction(self):
        """Return the fraction of a GPU this client requires (suitable for
        Ray's num_gpus argument).

        Returns:
            (mem, sm) requirements as a fraction of GPU capacity
        """
        props = properties.getProperties()
        reqs = props.resourceReqs(self.modelSpec.name, self.modelSpec.modelType)

        memFrac = reqs['mem'] / maxMemResource
        smFrac = reqs['sm'] / maxSMResource

        return (roundMIG(memFrac), roundMIG(smFrac))


class ServerState(enum.Enum):
    REGISTER = enum.auto()  # Waiting for clients to register
    WARM = enum.auto()  # Waiting for clients to warm themselves
    RUNNING = enum.auto()  # Actively running regular requests


class GPUAssignment():
    def __init__(self):
        self.clients = []
        self.resUtil = 0.0
        self.nWorkers = 0

    def assignWorker(self, client: clientState):
        self.clients.append(client)
        self.resUtil += client.resFrac
        self.nWorkers += 1


def checkForMPS():
    """Check if the MPS daemon is running"""
    for proc in psutil.process_iter():
        if 'nvidia-cuda-mps-control' in proc.cmdline():
            return True
    return False


def getSmallestGPU(gpus, requestedFrac):
    """Returns the least thread utilized GPU that can support requestedFrac
    new resources. If there is a tie, return the GPU with the least
    resource utilization."""
    minGPU = gpus[0]
    for gpu in gpus:
        if gpu.resUtil + requestedFrac > 1:
            continue

        if gpu.nWorkers < minGPU.nWorkers:
            minGPU = gpu
        elif gpu.resUtil < minGPU.resUtil:
            minGPU = gpu

    if minGPU.resUtil + requestedFrac > 1:
        return None
    else:
        return minGPU


class serverLoop():
    """ServerTask"""
    def __init__(self, clientSock, barrierSock, benchConfig):
        self.state = ServerState.REGISTER

        self.loop = IOLoop.instance()
        self.benchConfig = benchConfig

        self.clientStream = ZMQStream(clientSock)
        self.clientStream.on_recv(self.handleClients)

        self.barrierStream = ZMQStream(barrierSock)
        self.barrierStream.on_recv(self.handleBarrier)
        self.readyClients = []

        IOLoop.current().add_callback(self.handleWorker)

        self.nGPUs = infbench.getNGpu()

        self.pendingRegistrations = []

        self.clients = {}
        self.clientStats = profiling.profCollection()

        self.pool = kaas.pool.Pool(self.nGPUs, policy=benchConfig['policy'])

        self.rayQ = ray.util.queue.Queue()

        self.nOutstanding = 0
        self.maxOutstanding = kaas.pool.maxOutstanding
        self.overwhelmed = False

    def handleBarrier(self, msg):
        clientID = msg[0].decode('utf-8')

        print("Recieved Ready from: ", clientID)
        self.readyClients.append(clientID)
        if len(self.readyClients) == self.benchConfig['numClient']:
            # Get cold-start stats (if any) and reset for main warm passes
            poolStats = self.pool.getProfile()
            self.coldStats = self.clientStats
            self.coldStats.merge(poolStats)
            self.clientStats = profiling.profCollection()

            print("Releasing Barrier")
            for cID in self.readyClients:
                self.barrierStream.send_multipart([cID.encode('utf-8'), b'', b'GO'])

    async def handleWorker(self):
        result, reqData = await self.rayQ.get_async()
        clientID = reqData[0]
        reqID = reqData[1]

        result = flattenRayRefs(result)

        outBufs = [clientID.encode('utf-8'), reqID]
        outBufs.extend(result)
        self.clientStream.send_multipart(outBufs)
        IOLoop.current().add_callback(self.handleWorker)

        self.nOutstanding -= 1
        # Start accepting work again once we get below our max (plus some
        # hysteresis)
        if self.overwhelmed and self.nOutstanding < (self.maxOutstanding * 0.8):
            self.clientStream.on_recv(self.handleClients)

    def _assignClients(self, gpus, clients, maxNewWorkers):
        """Run one round of assignments of clients to GPUs. Returns the number
        of new workers assigned. Client workers are assigned to the GPU with
        the least number of workers if feasible, otherwise it goes to the GPU
        with the most free resources.
        Arguments:
            GPUs: heap of GPUAssignment
            clients: Map of {clientID: clientState}
            maxNewWorkers: Maximum number of new workers to create
        Returns:
            Number of new workers assigned
        """
        nNewWorker = 0
        for client in clients.values():
            if maxNewWorkers is not None and nNewWorker == maxNewWorkers:
                return nNewWorker

            minGPU = getSmallestGPU(gpus, client.resFrac)
            if minGPU is None:
                # No feasible solution found
                return nNewWorker
            else:
                minGPU.assignWorker(client)
                nNewWorker += 1
                client.nWorker += 1

        return nNewWorker

    def registerClients(self):
        if self.benchConfig['fractional'] is None and checkForMPS():
            print("WARNING: MPS is enabled in non-fractional mode")

        if self.benchConfig['policy'] == 'static':
            if self.benchConfig['fractional'] is None:
                resource = 'full'
            else:
                resource = self.benchConfig['fractional']

            if resource != 'full' and not checkForMPS():
                raise ServerError("MPS Daemon not running, static with a fractional argument requires the MPS daemon. Start with 'nvidia-cuda-mps-control -d'")

            # workers are assumed to consume one CPU (default for Ray). We need
            # to have at least one CPU free for tasks plus one for the pool. We
            # also can't allocate a worker that's less than 1/8 of a GPU.
            maxWorkers = min(psutil.cpu_count() - 2, self.nGPUs * 8)

            for clientID, modelName, modelType in self.pendingRegistrations:
                self.clients[clientID] = clientState(clientID, modelName, modelType,
                                                     fractional=resource)

            nWorker = 0
            GPUs = [GPUAssignment() for i in range(self.nGPUs)]

            # Everyone has to get at least one Worker
            nWorker += self._assignClients(GPUs, self.clients, maxWorkers)
            if nWorker < len(self.clients):
                raise ServerError("Cannot register all clients: insufficient resources")

            # Keep going until all GPUs have at least one client or we exceed
            # the maximum number of workers/exhuast resources. This policy
            # maximizes the minimum nThread per worker while remaining
            # work-conserving. However, it creates the minimum number of
            # workers which can lead to poor utilization when there are many
            # clients or clients with very different compute requirements. It
            # also may give some clients more threads and target resources than
            # others.
            target = min(maxWorkers, self.nGPUs)
            while nWorker < target:
                nWorker += self._assignClients(GPUs, self.clients, target - nWorker)

            # List of workers to create for each client (defined by resourc
            # requirements)
            # {client: [(threadFrac, resFrac)]}
            clientWorkers = collections.defaultdict(list)
            for gpu in GPUs:
                clientThreadFrac = roundMIG(1 / gpu.nWorkers, ceil=False)
                for client in gpu.clients:
                    clientWorkers[client.clientID].append((clientThreadFrac, client.resFrac))

            for clientID, resReqs in clientWorkers.items():
                self.pool.registerGroup(clientID,
                                        runActor,
                                        workerResources=resReqs)

        else:
            for clientID, modelName, modelType in self.pendingRegistrations:
                cState = clientState(clientID, modelName, modelType)
                self.clients[clientID] = cState
                self.pool.registerGroup(clientID, runActor)

    def handleClients(self, msg):
        clientID = msg[0].decode('utf-8')
        reqID = msg[1]
        data = msg[2:]

        if self.state == ServerState.REGISTER:
            modelName = reqID.decode('utf-8')
            modelType = data[0].decode('utf-8')
            self.pendingRegistrations.append((clientID, modelName, modelType))
            if len(self.pendingRegistrations) == self.benchConfig['numClient']:
                self.registerClients()
                print("All clients registered, entering warming phase.")
                for cID in self.clients.keys():
                    self.clientStream.send_multipart([cID.encode('utf-8'), b'', b'REGISTERED'])

                self.state = ServerState.WARM
        else:
            cState = self.clients[clientID]

            self.nOutstanding += 1

            # Too many outstanding queries can overwhelm Ray and hurt
            # throughput.
            if self.nOutstanding > self.maxOutstanding:
                self.clientStream.stop_on_recv()
                self.overwhelmed = True

            if cState.modelSpec.cacheInputs:
                idx = int.from_bytes(data[0], sys.byteorder)
                if idx not in cState.cachedInputs:
                    inputs = cState.loader.get(idx)
                    cState.cachedInputs[idx] = [ray.put(val) for val in inputs]
                inpRefs = cState.cachedInputs[idx]
            else:
                inpRefs = [ray.put(val) for val in data]

            _runOne(cState.modelSpec, cState.specRef, cState.modelArg,
                    cState.constRefs, inpRefs, completionQ=self.rayQ,
                    queryId=(clientID, reqID), clientID=clientID,
                    cacheModel=self.benchConfig['forceCold'],
                    inline=self.benchConfig['inline'], runPool=self.pool,
                    stats=self.clientStats.mod('groups').mod(clientID))

    def shutdown(self):
        self.clientStream.stop_on_recv()
        IOLoop.instance().stop()

        poolStats = self.pool.getProfile()
        self.warmStats = profiling.profCollection()
        self.warmStats.mod('server').merge(self.clientStats)
        self.warmStats.merge(poolStats)
        self.clientStats = profiling.profCollection()


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

    print("Reporting server stats:")
    resPath = "server_stats.json"
    print(f"Saving results to: {resPath}")
    infbench.saveReport(looper.warmStats, None, benchConfig, resPath)
