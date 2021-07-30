import ray
import ray.util.queue
import infbench
import threading
import time
import numpy as np
import os
import pickle
import signal
import subprocess as sp

from tornado.ioloop import IOLoop
import zmq
from zmq.eventloop.zmqstream import ZMQStream

import mlperf_loadgen
import libff.kaas.kaasRay as kaasRay

import util


# All steps (pre/run/post) take in multiple arguments (even if there's one
# argument, it's passed as a tuple). If we passed a list of futures, we would
# need to ray.get() each one seperately. This would prevent Ray from doing full
# lazy evaluation, it would instantiate a million functions each waiting on
# ray.get(), wasting a ton of resources and eventually crashing. Instead, we
# pass each input directly as an argument using the *batch syntax (this groups
# remaining function arguments into a list). This way Ray waits until all
# inputs are ready before instantiating the function.


def getNGpu():
    """Returns the number of available GPUs on this machine"""
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    else:
        proc = sp.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                      stdout=sp.PIPE, text=True, check=True)
        return proc.stdout.count('\n')


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


def _run(model, inputs, completionQ, queryId):
    """Internal run function"""
    constants, data = _unMarshalArgs(model.runMap, inputs)

    results = model.run(constants + list(data))

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
        if pid in nodeCache:
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
    constants, data = _unMarshalArgs(mClass.postMap, inputs)

    # KaaS will place the result directly into the object store and return a
    # reference to it. It should be fine to ray.get this reference because the
    # data is already in the obj store before we get called here. Other inputs
    # (e.g. from pre()) will already be dereferenced by ray.
    if modelSpec.modelType == "kaas":
        data = list(data)
        for i in range(len(data)):
            if isinstance(data[i], ray._raylet.ObjectRef):
                data[i] = ray.get(data[i])

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

    preInp = _getInputs(model.preMap, const=constants, inp=inputs)
    preOut = model.pre(preInp)

    runInp = _getInputs(model.runMap, const=constants, inp=inputs, pre=preOut)
    modOut = model.run(runInp)

    if model.noPost:
        postOut = modOut
    else:
        postInp = _getInputs(model.postMap, const=constants, inp=inputs,
                             pre=preOut, run=modOut)
        postOut = model.post(postInp)

    if completionQ is not None:
        completionQ.put((postOut, queryId))

    return postOut


def _getInputs(maps, const=None, inp=None, pre=None, run=None):
    inputs = []
    for (argMap, data) in zip(maps, [const, inp, pre, run]):
        if argMap is not None:
            assert data is not None
            inputs.extend([data[i] for i in argMap])
    return inputs


@ray.remote(num_gpus=1)
class runActor():
    """A persistent actor for running model requests. Actors will cache models
    as needed and run them natively. It is possible to run out of GPU memory
    with actors since they cache every model they are passed."""
    def __init__(self):
        self.modelCache = {}

    def run(self, modelSpec, modelArg, *inputs, completionQ=None, queryId=None,
            cacheModel=False, clientID=None):
        # The runActor must cache the model, if you wan't to reset, you must
        # kill and restart the actor. cacheModel is kept for consistency with
        # runTask but is ignored here.
        if clientID in self.modelCache:
            model = self.modelCache[clientID]
        else:
            model = modelSpec.modelClass(modelArg)
            self.modelCache[clientID] = model

        return _run(model, inputs, completionQ, queryId)

    def runKaas(self, req, queryId=None, completionQ=None):
        results = kaasRay.kaasServeRay(req.toDict())

        if completionQ is not None:
            completionQ.put((results, queryId))

        return results


class runnerPool():
    def __init__(self, nRunner):
        self.actors = []
        for i in range(nRunner):
            self.actors.append(runActor.remote())

        self.policy = self._roundRobin
        self.last = 0

    def _roundRobin(self, clientID):
        """A simple round-robin policy with no affinity"""
        self.last = (self.last + 1) % len(self.actors)
        return self.last

    def getRunner(self, clientID):
        """Return an actor suitable for running a request from clientID.
        Callers should call the returned actor exactly once per call to
        getRunner (do not cache the return value of getRunner)."""
        runnerIdx = self.policy(clientID)
        return self.actors[runnerIdx]


def _runOne(modelSpec, specRef, modelArg, constRefs, inputRefs, inline=False,
            completionQ=None, queryId=None, cacheModel=False, clientID=False,
            runPool=None):
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
        preInp = _getInputs(mClass.preMap, const=constRefs, inp=inputRefs)

        preOut = pre.options(num_returns=mClass.nOutPre).remote(specRef, *preInp)
        if mClass.nOutPre == 1:
            preOut = [preOut]

        # Run
        runInp = _getInputs(mClass.runMap, const=constRefs, inp=inputRefs, pre=preOut)

        if modelSpec.modelType == "kaas":
            model = modelArg
            req = model.run(runInp)

            if runPool is not None:
                runActor = runPool.getRunner(clientID)
                runner = runActor.runKaas.options(num_returns=mClass.nOutRun).remote
            else:
                runner = runKaasTask.options(num_returns=mClass.nOutRun).remote

            if completionQ is not None and mClass.noPost:
                runOut = runner(req, queryId=queryId, completionQ=completionQ)
            else:
                runOut = runner(req)
        else:
            if runPool is not None:
                runActor = runPool.getRunner(clientID)
                runner = runActor.run.options(num_returns=mClass.nOutRun).remote
            else:
                runner = runTask.options(num_returns=mClass.nOutRun).remote

            if completionQ is not None and mClass.noPost:
                runOut = runner(specRef, modelArg, *runInp,
                                completionQ=completionQ, queryId=queryId,
                                cacheModel=cacheModel, clientID=clientID)
            else:
                runOut = runner(specRef, modelArg, *runInp, cacheModel=cacheModel)

        if mClass.nOutRun == 1:
            runOut = [runOut]

        # Post
        if mClass.noPost:
            postOut = runOut
        else:
            postInp = _getInputs(mClass.postMap, const=constRefs,
                                 inp=inputRefs, pre=preOut, run=runOut)
            postOut = post.options(num_returns=mClass.nOutPost) \
                .remote(specRef, *postInp, completionQ=completionQ, queryId=queryId)

            if mClass.nOutPost == 1:
                postOut = [postOut]

    return postOut


def nShot(modelSpec, n, inline=False, useActors=False):
    ray.init()

    specRef = ray.put(modelSpec)

    modelArg = modelSpec.getModelArg()
    loader = modelSpec.loader(modelSpec.dataDir)

    loader.preLoad(range(min(n, loader.ndata)))
    constants = modelSpec.modelClass.getConstants(modelSpec.modelPath.parent)
    if constants is None:
        constRefs = None
    else:
        constRefs = []
        for const in constants:
            constRefs.append(ray.put(const))

    if useActors:
        pool = runnerPool(getNGpu())
    else:
        pool = None

    times = []
    accuracies = []
    results = []
    for i in range(n):
        idx = i % loader.ndata
        inp = loader.get(idx)

        start = time.time()
        res = _runOne(modelSpec, specRef, modelArg, constRefs, inp,
                      inline=inline, runPool=pool)
        res = ray.get(res)

        results.append(res)

        times.append(time.time() - start)

        if loader.checkAvailable:
            accuracies.append(loader.check(res, idx))

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
        resp, qid = queue.get()

        # if ncomplete % 10 == 0:
        #     print("Query {} Finished".format(ncomplete))

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
            completion = mlperf_loadgen.QuerySampleResponse(qid, 0, 0)
            mlperf_loadgen.QuerySamplesComplete([completion])
            ncomplete += 1


class mlperfRunner():
    def __init__(self, modelSpec, loader, constantRefs, inline=False, useActors=False):
        self.modelSpec = modelSpec
        self.modelArg = modelSpec.getModelArg()
        self.loader = loader
        self.constants = constantRefs
        self.inline = inline

        # Total number of queries issued
        self.nIssued = 0

        self.specRef = ray.put(self.modelSpec)

        self.nGpu = getNGpu()

        if useActors:
            self.pool = runnerPool(self.nGpu)
        else:
            self.pool = None

    def start(self, preWarm=True):
        self.completionQueue = ray.util.queue.Queue()
        self.completionHandler = threading.Thread(
                target=handleCompletion, args=[self.completionQueue])
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
                               self.constants, inputs, inline=self.inline,
                               runPool=self.pool))
            for res in results:
                ray.get(res)

    def runBatch(self, queryBatch):
        for q in queryBatch:
            inp = self.loader.get(q.index)

            _runOne(self.modelSpec, self.specRef, self.modelArg,
                    self.constants, inp, inline=self.inline,
                    completionQ=self.completionQueue, queryId=q.id,
                    cacheModel=True, runPool=self.pool)

        self.nIssued += len(queryBatch)

    def stop(self):
        self.completionQueue.put((self.nIssued, None))
        print("Waiting for completion handler to finish")
        self.completionHandler.join()


def mlperfBench(modelSpec, testing=False, inline=False, useActors=False):
    """Run the mlperf loadgen version"""
    ray.init()

    settings = modelSpec.modelClass.getMlPerfCfg(testing=testing)
    loader = modelSpec.loader(modelSpec.dataDir)

    constants = modelSpec.modelClass.getConstants(modelSpec.modelPath.parent)
    if constants is None:
        constRefs = None
    else:
        constRefs = []
        for const in constants:
            constRefs.append(ray.put(const))

    runner = mlperfRunner(modelSpec, loader, constRefs, inline=inline, useActors=useActors)

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


# =============================================================================
# Server Mode
# =============================================================================


class clientState():
    def __init__(self, modelName):
        self.modelSpec = util.getModelSpec(modelName)
        self.specRef = ray.put(self.modelSpec)
        self.modelArg = self.modelSpec.getModelArg()

        constants = self.modelSpec.modelClass.getConstants(self.modelSpec.modelPath.parent)
        if constants is None:
            constRefs = None
        else:
            constRefs = []
            for const in constants:
                constRefs.append(ray.put(const))
        self.constRefs = constRefs


# { clientID -> clientState }
# For now, we only support one model per client
clients = {}


class serverLoop():
    """ServerTask"""
    def __init__(self, clientSock, useActors=False):
        self.loop = IOLoop.instance()

        self.clientStream = ZMQStream(clientSock)
        self.clientStream.on_recv(self.handleClients)

        IOLoop.current().add_callback(self.handleWorker)

        self.nGpu = getNGpu()

        if useActors:
            self.pool = runnerPool(self.nGpu)
        else:
            self.pool = None

        self.rayQ = ray.util.queue.Queue()

    async def handleWorker(self):
        result, reqData = await self.rayQ.get_async()
        clientID = reqData[0]
        reqID = reqData[1]

        self.clientStream.send_multipart([clientID, reqID, pickle.dumps(result)])
        IOLoop.current().add_callback(self.handleWorker)

    def handleClients(self, msg):
        clientID, reqID, data = msg

        cState = clients.get(clientID, None)

        # Registration
        if cState is None:
            print("Registering ", clientID)
            modelName = reqID.decode('utf-8')
            cState = clientState(modelName)
            clients[clientID] = cState
        else:
            # XXX ray.put is just going to re-pickle the data. We should really
            # require models to only pass bytes as inputs and outputs.
            data = pickle.loads(data)

            _runOne(cState.modelSpec, cState.specRef, cState.modelArg,
                    cState.constRefs, data, completionQ=self.rayQ,
                    queryId=(clientID, reqID), clientID=clientID,
                    cacheModel=True, inline=False, runPool=self.pool)

    def shutdown(self):
        self.clientStream.stop_on_recv()
        IOLoop.instance().stop()


def serveRequests(useActors=False):
    ray.init()
    context = zmq.Context()

    clientSock = context.socket(zmq.ROUTER)
    clientSock.bind(util.clientUrl)

    # IOLoop uses a global context, when you instantiate a serverLoop object,
    # it registers itself with IOLoop. The returned object isn't used here.
    looper = serverLoop(clientSock, useActors=useActors)

    signal.signal(signal.SIGINT, lambda s, f: IOLoop.instance().add_callback_from_signal(looper.shutdown))

    print("Beginning serving loop")
    IOLoop.instance().start()
    print("Server Exiting")
