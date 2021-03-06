#!/usr/bin/env python

import sys
import threading
from tornado.ioloop import IOLoop
import zmq
from zmq.eventloop.zmqstream import ZMQStream
import time
import random
from pprint import pprint

import mlperf_loadgen

import infbench
from infbench import properties
from kaas import profiling
import util

sutSockUrl = "inproc://sut"


def setupZmq(clientID, context=None):
    if context is None:
        context = zmq.Context()
    serverSocket = context.socket(zmq.DEALER)
    serverSocket.identity = clientID
    serverSocket.connect(util.clientUrl)

    barrierSocket = context.socket(zmq.REQ)
    barrierSocket.identity = clientID
    barrierSocket.connect(util.barrierUrl)

    return serverSocket, barrierSocket


def barrier(barrierSock):
    barrierSock.send(b'READY')
    barrierSock.recv()


def sendReq(socket, clientID, data):
    payload = [clientID]
    payload.extend(data)
    socket.send_multipart(payload)


def preWarm(serverSock, barrierSock, inputs):
    nReq = infbench.getNGpu()*2
    for i in range(nReq):
        sendReq(serverSock, bytes(1), inputs)
    for i in range(nReq):
        serverSock.recv_multipart()

    barrier(barrierSock)


def register(serverSock, barrierSock, benchConfig):
    sendReq(serverSock,
            benchConfig['model'].encode('utf-8'),
            [benchConfig['modelType'].encode('utf-8')])
    serverSock.recv_multipart()


# =============================================================================
# nShot
# =============================================================================

def _nShotSync(inpIds, loader, serverSocket, stats=None, cacheInputs=False):
    results = []
    stats['n_req'].increment(len(inpIds))

    with profiling.timer("t_all", stats):
        for reqId, idx in zip(range(len(inpIds)), inpIds):
            if cacheInputs:
                inp = [idx.to_bytes(8, sys.byteorder)]
            else:
                inp = loader.get(idx)

            with profiling.timer('t_e2e', stats):
                sendReq(serverSocket, reqId.to_bytes(4, sys.byteorder), inp)

                resp = serverSocket.recv_multipart()
                respId = int.from_bytes(resp[0], sys.byteorder)

                assert (respId == reqId)
                results.append((idx, resp[1:]))

    return results


def _nShotAsync(inpIds, loader, serverSocket, stats=None, cacheInputs=False):
    results = []

    # reqId -> (startTime, inpId)
    requests = {}
    with profiling.timer('t_all', stats):
        # for i in range(n):
        for reqId, idx in zip(range(len(inpIds)), inpIds):
            if cacheInputs:
                inp = [idx.to_bytes(8, sys.byteorder)]
            else:
                inp = loader.get(idx)

            requests[reqId] = (time.time(), idx)
            sendReq(serverSocket, reqId.to_bytes(4, sys.byteorder), inp)

        stats['n_req'].increment(len(inpIds))

        for i in range(len(inpIds)):
            resp = serverSocket.recv_multipart()

            respIdx = int.from_bytes(resp[0], sys.byteorder)

            respStartTime, respInpId = requests[respIdx]
            stats['t_e2e'].increment((time.time() - respStartTime)*1000)

            # Server is permitted to return requests out of order
            assert (respInpId in inpIds)

            results.append((respInpId, resp[1:]))

    return results


def nShot(modelSpec, n, benchConfig):
    """A simple nShot test. All N requests are sent before waiting for
    responses. The raw results are returned."""
    clientID = benchConfig['name'].encode('utf-8')

    coldStats = profiling.profCollection(detail=True)
    warmStats = profiling.profCollection(detail=True)

    nWarmStep = infbench.getNGpu()*2

    nLoad = max(n, nWarmStep)
    loader = modelSpec.loader(modelSpec.dataDir)
    # inpIds = [i % loader.ndata for i in range(nLoad)]
    inpIds = [random.randrange(0, loader.ndata) for i in range(nLoad)]
    loader.preLoad(inpIds)

    zmqContext = zmq.Context()
    serverSocket, barrierSocket = setupZmq(clientID, context=zmqContext)

    # Registration
    print("Registering client: ", benchConfig['name'])
    register(serverSocket, barrierSocket, benchConfig)

    # Cold starts
    _nShotAsync(inpIds[:nWarmStep], loader, serverSocket,
                stats=coldStats, cacheInputs=modelSpec.cacheInputs)

    print("Waiting for other clients:")
    barrier(barrierSocket)

    # Send all requests
    print("Sending Requests:")
    results = _nShotSync(inpIds[:n], loader, serverSocket,
                         stats=warmStats, cacheInputs=modelSpec.cacheInputs)
    print("Test complete")

    if loader.checkAvailable:
        print("Checking accuracy")
        accuracies = []
        for idx, res in results:
            accuracies.append(loader.check(res, idx))

        print("Accuracy = ", sum([int(res) for res in accuracies]) / n)
    else:
        print("Accuracy checking not supported for this model")

    report = warmStats.report(metrics=['mean'])

    print("Stats: ")
    pprint(report)

    print("Writing full report to: ", benchConfig['name'] + '_results.json')
    infbench.saveReport(warmStats, coldStats, benchConfig, benchConfig['name'] + '_results.json')

    return results


# =============================================================================
# Throughput
# =============================================================================

class throughputLoop():
    def __init__(self, modelSpec, benchConfig, zmqContext, targetTime=60):
        """This test uses tornado IOLoop to submit requests as fast as possible
        for targetTime seconds. It reports the total throughput acheived."""
        self.benchConfig = benchConfig
        self.modelSpec = modelSpec
        self.clientID = benchConfig['name'].encode('utf-8')
        self.loop = IOLoop.instance()

        self.loader = modelSpec.loader(modelSpec.dataDir)
        self.loader.preLoad(range(self.loader.ndata))

        # This number needs to be big enough to ensure that there's always
        # enough work on the server to get full bandwidth. It can't be too big
        # though because then the server will throttle everyone in order to
        # keep the pool happy, this might hurt load balancing. This isn't too
        # sensitive so long as it's reasonably large (nGPU*2 seems to work
        # well). Also, when nClient >> nGPU, load balancing isn't really an
        # issue anyway.
        self.targetOutstanding = infbench.getNGpu() * 2

        # Are we at targetOutstanding (and therefore not submitting requests)?
        self.qFull = False

        self.targetTime = targetTime
        self.nOutstanding = 0
        self.nextIdx = 0
        self.nCompleted = 0
        self.done = False

        self.serverSocket, self.barrierSocket = setupZmq(self.clientID, context=zmqContext)

        # Register and PreWarm
        register(self.serverSocket, self.barrierSocket, self.benchConfig)

        if self.modelSpec.cacheInputs:
            inputs = [(0).to_bytes(8, sys.byteorder)]
        else:
            loader = modelSpec.loader(modelSpec.dataDir)
            loader.preLoad([0])
            inputs = loader.get(0)

        preWarm(self.serverSocket, self.barrierSocket, inputs)

        # start listening for server responses
        self.serverStream = ZMQStream(self.serverSocket)
        self.serverStream.on_recv(self.handleServer)

        self.loop.add_callback(self.submitReqs)
        self.startTime = time.time()

    def submitReqs(self):
        while self.nOutstanding < self.targetOutstanding:
            idx = self.nextIdx % self.loader.ndata
            if self.modelSpec.cacheInputs:
                inp = [idx.to_bytes(8, sys.byteorder)]
            else:
                inp = self.loader.get(idx)
            sendReq(self.serverStream, self.nextIdx.to_bytes(8, sys.byteorder), inp)
            self.nextIdx += 1
            self.nOutstanding += 1

        self.qFull = True

    def handleServer(self, msg):
        """Handle responses from the server"""
        # If we're done, we need to wait for any outstanding requests to
        # complete before shutting down (avoids various race conditions and
        # dirty state)
        if self.done:
            self.nOutstanding -= 1
            if self.nOutstanding == 0:
                self.serverStream.stop_on_recv()
                IOLoop.instance().stop()
            else:
                return

        self.nOutstanding -= 1
        self.nCompleted += 1

        # This is unlikely to trigger unless I did something wrong, especially
        # if targetOutstanding is way too low.
        if self.nOutstanding == 0:
            self.targetOutstanding += 1

        if time.time() - self.startTime >= self.targetTime:
            print("Test complete, shutting down")
            self.runTime = time.time() - self.startTime
            self.done = True
            if self.nOutstanding == 0:
                self.serverStream.stop_on_recv()
                IOLoop.instance().stop()

        elif self.nOutstanding < self.targetOutstanding and self.qFull:
            # qFull is needed because we might get multiple responses before
            # the submitReqs callback gets scheduled and we don't want to
            # submit multiple submitReqs callbacks.
            self.qFull = False
            self.loop.add_callback(self.submitReqs)

    def reportMetrics(self):
        metrics = profiling.profCollection()

        # useful for debugging mostly. Ideally t_total ~= targetTime
        metrics['n_completed'].increment(self.nCompleted)
        metrics['t_total'].increment(self.runTime * 1000)  # we always report time in ms

        # completions/second
        metrics['throughput'].increment(self.nCompleted / self.runTime)

        if self.nCompleted < self.targetOutstanding:
            print("\n*********************************************************")
            print("WARNING: Too few queries completed!")
            print("\tnQuery=", self.nCompleted)
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

        return metrics, valid


def throughput(modelSpec, benchConfig):
    context = zmq.Context()

    if benchConfig['runTime'] is None:
        runTime = 300
    else:
        runTime = benchConfig['runTime']

    testLoop = throughputLoop(modelSpec, benchConfig, context, targetTime=runTime)
    IOLoop.instance().start()

    metrics, valid = testLoop.reportMetrics()
    benchConfig['valid'] = valid

    infbench.saveReport(metrics, None, benchConfig, benchConfig['name'] + '_results.json')


# =============================================================================
# MlPerf
# =============================================================================

class mlperfRunner(threading.Thread):
    def __init__(self, modelSpec, benchConfig, zmqContext):
        self.zmqContext = zmqContext
        self.benchConfig = benchConfig
        self.modelSpec = modelSpec
        self.metrics = profiling.profCollection(detail=True)
        self.nQuery = 0

        threading.Thread.__init__(self)

    def runBatch(self, queryBatch):
        for q in queryBatch:
            self.nQuery += 1
            if self.modelSpec.cacheInputs:
                inp = [q.index.to_bytes(8, sys.byteorder)]
            else:
                inp = self.loader.get(q.index)
            sendReq(self.sutSock, q.id.to_bytes(8, sys.byteorder), inp)

    def processLatencies(self, latencies):
        self.metrics['t_response'] = infbench.processLatencies(self.benchConfig, latencies)

    def run(self):
        self.loader = self.modelSpec.loader(self.modelSpec.dataDir)

        self.sutSock = self.zmqContext.socket(zmq.PAIR)
        self.sutSock.connect(sutSockUrl)

        props = properties.getProperties()
        runSettings = props.getMlPerfConfig(self.modelSpec.name, self.benchConfig)

        logSettings = mlperf_loadgen.LogSettings()
        logSettings.log_output.prefix = self.benchConfig['name'] + "_"

        sut = mlperf_loadgen.ConstructSUT(
            self.runBatch, infbench.model.flushQueries, self.processLatencies)

        qsl = mlperf_loadgen.ConstructQSL(
            self.loader.ndata, infbench.model.mlperfNquery, self.loader.preLoad, self.loader.unLoad)

        start = time.time()
        mlperf_loadgen.StartTestWithLogSettings(sut, qsl, runSettings, logSettings)
        self.metrics['t_e2e'].increment(time.time() - start)

        mlperf_loadgen.DestroyQSL(qsl)
        mlperf_loadgen.DestroySUT(sut)

        # Shutdown Signal
        self.sutSock.send_multipart([b'', b''])
        self.sutSock.close()

        self.metrics['n_query'].increment(self.nQuery)


class mlperfLoop():
    def __init__(self, modelSpec, benchConfig, zmqContext):
        self.benchConfig = benchConfig
        self.modelSpec = modelSpec
        self.clientID = benchConfig['name'].encode('utf-8')
        self.loop = IOLoop.instance()

        self.sutSocket = zmqContext.socket(zmq.PAIR)
        self.sutSocket.bind(sutSockUrl)
        self.sutStream = ZMQStream(self.sutSocket)
        self.sutStream.on_recv(self.handleSut)

        self.serverSocket, self.barrierSocket = setupZmq(self.clientID, context=zmqContext)
        self.serverStream = ZMQStream(self.serverSocket)
        self.serverStream.on_recv(self.handleServer)

        # Register with the benchmark server
        register(self.serverSocket, self.barrierSocket, self.benchConfig)

        # PreWarm
        if self.modelSpec.cacheInputs:
            inputs = [(0).to_bytes(8, sys.byteorder)]
        else:
            loader = modelSpec.loader(modelSpec.dataDir)
            loader.preLoad([0])
            inputs = loader.get(0)

        preWarm(self.serverSocket, self.barrierSocket, inputs)

    def handleSut(self, msg):
        """Handle requests from mlperf, we simply proxy requests to serverSocket"""
        if msg[0] == b'':
            print("Test complete, shutting down")
            self.sutStream.stop_on_recv()
            self.serverStream.stop_on_recv()
            IOLoop.instance().stop()
        else:
            self.serverStream.send_multipart(msg)

    def handleServer(self, msg):
        """Handle responses from the server"""
        # reqID, data = msg
        reqID = msg[0]

        # We use a 1-byte 0 reqID for non mlperf queries
        if len(reqID) == 8:
            qID = int.from_bytes(reqID, sys.byteorder)
            completion = mlperf_loadgen.QuerySampleResponse(qID, 0, 0)
            mlperf_loadgen.QuerySamplesComplete([completion])


def mlperfBench(modelSpec, benchConfig):
    context = zmq.Context()

    print("Registering and prewarming", modelSpec.name)
    mlperfLoop(modelSpec, benchConfig, context)

    testRunner = mlperfRunner(modelSpec, benchConfig, context)

    print("Beginning loadgen run")
    testRunner.start()

    IOLoop.instance().start()
    mlPerfMetrics, valid = infbench.parseMlPerf(benchConfig['name'] + '_')
    benchConfig['valid'] = valid

    metrics = profiling.profCollection(detail=True)
    metrics.merge(testRunner.metrics)
    metrics.merge(mlPerfMetrics)

    infbench.saveReport(metrics, None, benchConfig, benchConfig['name'] + '_results.json')
