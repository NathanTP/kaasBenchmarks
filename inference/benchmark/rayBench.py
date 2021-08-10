import ray
import ray.util.queue
import infbench
import threading
import os
import pickle
import signal
import subprocess as sp
from pprint import pprint
import pathlib
import json
import collections

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
        return ([], list(args))
    else:
        nConst = len(argMap.const)
        constants = list(args[:nConst])
        inputs = list(args[nConst:])
        return (constants, inputs)


def handleKaasResult(res):
    # KaaS will place the result directly into the object store and return a
    # reference to it. It should be fine to ray.get this reference because the
    # data is already in the obj store before we get called here. Other inputs
    # (e.g. from pre()) will already be dereferenced by ray.

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
    constants, data = _unMarshalArgs(mClass.postMap, inputs)

    if modelSpec.modelType == 'kaas':
        data = handleKaasResult(data)

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

    def runNative(self, modelSpec, modelArg, *inputs, completionQ=None, queryId=None,
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

    def terminate(self):
        ray.actor.exit_actor()


class runnerPool():
    def __init__(self, nRunner, benchConfig):
        """RunnerPool is responsible for launching run requests.
                - nRunner: If using actors, number of actors to allocate
                - policy: Scheduling policy (when using actors)
                - mode: ['actors', 'kaas', 'task']. Actors and KaaS will run in
                actors while 'task' will use ray tasks instead.
        """
        self.maxRunners = nRunner
        self.mode = benchConfig['runner_mode']
        benchConfig['runner_policy']

        if self.mode not in ['task', 'actor', 'kaas']:
            raise ValueError("Unrecognized mode: " + self.mode)

        if self.mode != 'task':
            if benchConfig['runner_policy'] == 'rr':
                self.policy = self._roundRobin
                self.last = 0
                self.actors = []
                for i in range(nRunner):
                    self.actors.append(runActor.remote())
            elif benchConfig['runner_policy'] == 'affinity':
                self.policy = self._affinity
                self.lru = collections.deque()
                self.assignments = {}
            else:
                raise ValueError("Unrecognized policy: " + benchConfig['runner_policy'])

    def _roundRobin(self, clientID):
        """A simple round-robin policy with no affinity"""
        self.last = (self.last + 1) % len(self.actors)
        return self.actors[self.last]

    def _affinity(self, clientID):
        """Maps clients to actors exclusively. If there are not enough actors,
        one is killed to make room. This is most similar to a typical naive
        faas system."""
        if clientID in self.assignments:
            self.lru.remove(clientID)
            self.lru.appendleft(clientID)
            return self.assignments[clientID]
        else:
            if len(self.assignments) < self.maxRunners:
                self.lru.appendleft(clientID)
                self.assignments[clientID] = runActor.remote()
                return self.assignments[clientID]
            else:
                # Gotta evict someone
                idEvict = self.lru.pop()
                actEvict = self.assignments[idEvict]
                actEvict.terminate.remote()
                del self.assignments[idEvict]

                self.lru.appendleft(clientID)
                self.assignments[clientID] = runActor.remote()
                return self.assignments[clientID]

    def _getRunner(self, clientID):
        """Return an actor suitable for running a request from clientID.
        Callers should call the returned actor exactly once per call to
        getRunner (do not cache the return value of getRunner)."""
        return self.policy(clientID)

    def run(self, nReturn, clientID, args, kwargs={}):
        """Run a model. Args and kwargs will be passed to the appropriate runner"""
        if self.mode == 'task':
            return runTask.options(num_returns=nReturn).remote(*args, **kwargs)
        elif self.mode == 'actor':
            runActor = self._getRunner(clientID)
            respFutures = runActor.runNative.options(num_returns=nReturn).remote(*args, **kwargs)
        elif self.mode == 'kaas':
            runActor = self._getRunner(clientID)
            respFutures = runActor.runKaas.options(num_returns=nReturn).remote(*args, **kwargs)

        return respFutures


def _runOne(modelSpec, specRef, modelArg, constRefs, inputRefs, inline=False,
            completionQ=None, queryId=None, cacheModel=False, clientID=None,
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

            if completionQ is not None and mClass.noPost:
                runOut = runPool.run(mClass.nOutRun, clientID, [req], {"queryId": queryId, "completionQ": completionQ})
            else:
                runOut = runPool.run(mClass.nOutRun, clientID, [req])
        else:
            if completionQ is not None and mClass.noPost:
                runOut = runPool.run(mClass.nOutRun, clientID,
                                     [specRef, modelArg] + runInp,
                                     {"completionQ": completionQ, "queryId": queryId,
                                      "cacheModel": cacheModel, "clientID": clientID})
            else:
                runOut = runPool.run(mClass.nOutRun, clientID, [specRef, modelArg] + runInp, {"cacheModel": cacheModel})

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


def nShot(modelSpec, n, benchConfig, reportPath="results.json"):
    ray.init()

    stats = util.profCollection()

    specRef = ray.put(modelSpec)

    with util.timer("t_registerModel", stats):
        if modelSpec.modelType == "kaas":
            modelArg = modelSpec.getModelArg()
        else:
            modelArg = ray.put(modelSpec.getModelArg())

        constants = modelSpec.modelClass.getConstants(modelSpec.modelPath.parent)

        if constants is None:
            constRefs = None
        else:
            constRefs = []
            for const in constants:
                constRefs.append(ray.put(const))

    with util.timer("t_initLoader", stats):
        loader = modelSpec.loader(modelSpec.dataDir)
        loader.preLoad(range(min(n, loader.ndata)))

    pool = runnerPool(getNGpu(), benchConfig)

    accuracies = []
    results = []
    for i in range(n):
        idx = i % loader.ndata
        inp = loader.get(idx)

        with util.timer('t_e2e', stats):
            # Ray is lazy and asynchronous so it's difficult to collect more
            # detailed metrics than e2e. Details within the remote functions
            # should match localBench results anyway.
            res = _runOne(modelSpec, specRef, modelArg, constRefs, inp,
                          inline=benchConfig['inline'], runPool=pool,
                          cacheModel=benchConfig['cache'])

            res = ray.get(res)
            if modelSpec.modelType == 'kaas' and modelSpec.modelClass.noPost:
                res = handleKaasResult(res)

        results.append(res)

        if loader.checkAvailable:
            accuracies.append(loader.check(res, idx))

    if loader.checkAvailable:
        print("Accuracy = ", sum([int(res) for res in accuracies]) / n)
    else:
        print("Accuracy checking not supported by this dataset")

    report = stats.report()
    print("E2E Results:")
    pprint({(k, v) for (k, v) in report['t_e2e'].items() if k != "events"})

    if not isinstance(reportPath, pathlib.Path):
        reportPath = pathlib.Path(reportPath).resolve()

    print("Saving results to: ", reportPath)
    if reportPath.exists():
        with open(reportPath, 'r') as f:
            fullReport = json.load(f)
    else:
        fullReport = []

    record = {
        "config": benchConfig,
        "metrics": report
    }
    fullReport.append(record)

    with open(reportPath, 'w') as f:
        json.dump(fullReport, f)

    # print("Ray Profiling:")
    # ray.timeline(filename="rayProfile.json")

    return results


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

        if modelSpec.modelType == "kaas":
            self.modelArg = modelSpec.getModelArg()
        else:
            self.modelArg = ray.put(modelSpec.getModelArg())

        # Total number of queries issued
        self.nIssued = 0

        self.specRef = ray.put(self.modelSpec)

        self.nGpu = getNGpu()

        if benchConfig['actor_policy']:
            self.pool = runnerPool(self.nGpu, policy=benchConfig['actor_policy'])
        else:
            self.pool = None

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
                               runPool=self.pool))
            for res in results:
                ray.get(res)

    def runBatch(self, queryBatch):
        for q in queryBatch:
            inp = self.loader.get(q.index)

            _runOne(self.modelSpec, self.specRef, self.modelArg,
                    self.constants, inp, inline=self.benchConfig['inline'],
                    completionQ=self.completionQueue, queryId=q.id,
                    cacheModel=self.benchConfig['cache'], runPool=self.pool)

        self.nIssued += len(queryBatch)

    def processLatencies(self, latencies):
        self.latMetrics = infbench.model.processLatencies(self.benchConfig, latencies)

    def stop(self):
        self.completionQueue.put((self.nIssued, None))
        print("Waiting for completion handler to finish")
        self.completionHandler.join()


def mlperfBench(modelSpec, benchConfig):
    """Run the mlperf loadgen version"""
    ray.init()

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

    runner.stop()

    print("\nResults:")
    mlPerfMetrics = infbench.model.parseMlPerf('mlperf_log_')
    infbench.model.saveReport({**runner.latMetrics, **mlPerfMetrics}, benchConfig, 'results.json')


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
    def __init__(self, clientSock, barrierSock, benchConfig):
        self.loop = IOLoop.instance()
        self.benchConfig = benchConfig

        self.clientStream = ZMQStream(clientSock)
        self.clientStream.on_recv(self.handleClients)

        self.barrierStream = ZMQStream(barrierSock)
        self.barrierStream.on_recv(self.handleBarrier)
        self.readyClients = []

        IOLoop.current().add_callback(self.handleWorker)

        self.nGpu = getNGpu()

        if self.benchConfig['actor_policy'] is not None:
            self.pool = runnerPool(self.nGpu, policy=self.benchConfig['actor_policy'])
        else:
            self.pool = None

        self.rayQ = ray.util.queue.Queue()

    def handleBarrier(self, msg):
        clientID = msg[0]

        print("Recieved Ready from: ", clientID.decode("utf-8"))
        self.readyClients.append(clientID)
        if len(self.readyClients) == self.benchConfig['numClient']:
            print("Releasing Barrier")
            for cID in self.readyClients:
                self.barrierStream.send_multipart([cID, b'', b'GO'])

    async def handleWorker(self):
        result, reqData = await self.rayQ.get_async()
        clientID = reqData[0]
        reqID = reqData[1]

        self.clientStream.send_multipart([clientID, reqID, pickle.dumps(result)])
        IOLoop.current().add_callback(self.handleWorker)

    def handleClients(self, msg):
        clientID, reqID, data = msg

        cState = clients.get(clientID, None)

        if cState is None:
            # Registration
            print("Registering ", clientID)
            modelName = reqID.decode('utf-8')
            cState = clientState(modelName)
            clients[clientID] = cState
        else:
            # Normal Request

            # XXX ray.put is just going to re-pickle the data. We should really
            # require models to only pass bytes as inputs and outputs.
            data = pickle.loads(data)

            _runOne(cState.modelSpec, cState.specRef, cState.modelArg,
                    cState.constRefs, data, completionQ=self.rayQ,
                    queryId=(clientID, reqID), clientID=clientID,
                    cacheModel=self.benchConfig['cache'],
                    inline=self.benchConfig['inline'], runPool=self.pool)

    def shutdown(self):
        self.clientStream.stop_on_recv()
        IOLoop.instance().stop()


def serveRequests(benchConfig):
    ray.init()
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
