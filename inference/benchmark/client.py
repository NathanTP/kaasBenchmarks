#!/usr/bin/env python

import sys
import threading
from tornado.ioloop import IOLoop
import zmq
from zmq.eventloop.zmqstream import ZMQStream
import time

import mlperf_loadgen

import infbench
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


# =============================================================================
# nShot
# =============================================================================

def _nShotSync(n, loader, serverSocket, stats=None):
    results = []
    stats['n_req'].increment(n)

    with infbench.timer("t_all", stats):
        for i in range(n):
            idx = i % loader.ndata
            inp = loader.get(idx)

            with infbench.timer('t_e2e', stats):
                sendReq(serverSocket, idx.to_bytes(4, sys.byteorder), inp)

                resp = serverSocket.recv_multipart()
                respIdx = int.from_bytes(resp[0], sys.byteorder)

                assert (respIdx == idx)
                results.append((respIdx, resp[1:]))

    return results


def _nShotASync(n, loader, serverSocket, stats=None):
    results = []
    starts = []
    with infbench.timer('t_all', stats):
        for i in range(n):
            idx = i % loader.ndata
            inp = loader.get(idx)

            starts.append(time.time())
            sendReq(serverSocket, idx.to_bytes(4, sys.byteorder), inp)

        stats['n_req'].increment(n)

        for i in range(n):
            resp = serverSocket.recv_multipart()

            respIdx = int.from_bytes(resp[0], sys.byteorder)
            stats['t_e2e'].increment(time.time() - starts[respIdx])
            results.append((respIdx, resp[1:]))

    return results


def nShot(modelSpec, n, benchConfig):
    """A simple nShot test. All N requests are sent before waiting for
    responses. The raw results are returned."""
    clientID = benchConfig['name'].encode('utf-8')

    stats = infbench.profCollection()

    loader = modelSpec.loader(modelSpec.dataDir)
    loader.preLoad(range(min(n, loader.ndata)))

    zmqContext = zmq.Context()
    serverSocket, barrierSocket = setupZmq(clientID, context=zmqContext)

    # Registration
    print("Registering client: ", benchConfig['name'])
    sendReq(serverSocket, benchConfig['model'].encode('utf-8'), b'')

    # Cold starts
    inp = loader.get(0)
    for i in range(infbench.getNGpu()*2):
        sendReq(serverSocket, bytes(1), inp)
        serverSocket.recv_multipart()

    print("Waiting for other clients:")
    barrier(barrierSocket)

    # Send all requests
    print("Sending Requests:")
    results = _nShotSync(n, loader, serverSocket, stats=stats)
    print("Test complete")

    if loader.checkAvailable:
        print("Checking accuracy")
        accuracies = []
        for idx, res in results:
            accuracies.append(loader.check(res, idx))

        print("Accuracy = ", sum([int(res) for res in accuracies]) / n)
    else:
        print("Accuracy checking not supported for this model")

    report = stats.report()

    print("Stats: ")
    infbench.printReport(report, metrics=['mean'])

    print("Writing full report to: ", benchConfig['name'] + '_results.json')
    infbench.saveReport(report, None, benchConfig, benchConfig['name'] + '_results.json')

    return results


# =============================================================================
# Throughput
# =============================================================================

class throughputLoop():
    def __init__(self, modelSpec, benchConfig, zmqContext, targetTime=60):
        """This test uses tornado IOLoop to submit requests as fast as possible
        for targetTime seconds. It reports the total throughput acheived."""
        self.benchConfig = benchConfig
        self.clientID = benchConfig['name'].encode('utf-8')
        self.loop = IOLoop.instance()

        self.loader = modelSpec.loader(modelSpec.dataDir)
        self.loader.preLoad(range(self.loader.ndata))

        gpuType = infbench.getGpuType()
        maxQps, medianLat = modelSpec.modelClass.getPerfEstimates(gpuType)

        # This number is more sensitive than it should be. I'm not sure why but
        # setting it too high really hurts performance, even though the server
        # is throttled. 64 seems to work in practice.
        self.targetOutstanding = 64

        self.targetTime = targetTime
        self.nOutstanding = 0
        self.nextIdx = 0
        self.nCompleted = 0
        self.done = False

        self.serverSocket, self.barrierSocket = setupZmq(self.clientID, context=zmqContext)

        # Register and PreWarm
        sendReq(self.serverSocket, self.benchConfig['model'].encode('utf-8'), b'')
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
            inp = self.loader.get(self.nextIdx % self.loader.ndata)
            sendReq(self.serverStream, self.nextIdx.to_bytes(8, sys.byteorder), inp)
            self.nextIdx += 1
            self.nOutstanding += 1

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

        if time.time() - self.startTime >= self.targetTime:
            print("Test complete, shutting down")
            self.runTime = time.time() - self.startTime
            self.done = True

        self.nOutstanding -= 1
        self.nCompleted += 1

        if self.nOutstanding < (self.targetOutstanding / 2):
            self.loop.add_callback(self.submitReqs)

    def reportMetrics(self):
        metrics = infbench.profCollection()

        # useful for debugging mostly. Ideally t_total ~= targetTime
        metrics['n_completed'].increment(self.nCompleted)
        metrics['t_total'].increment(self.runTime * 1000)  # we always report time in ms

        # completions/second
        metrics['throughput'].increment(self.nCompleted / self.runTime)

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

    infbench.saveReport(metrics.report(), None, benchConfig, benchConfig['name'] + '_results.json')


# =============================================================================
# MlPerf
# =============================================================================

class mlperfRunner(threading.Thread):
    def __init__(self, modelSpec, benchConfig, zmqContext):
        self.zmqContext = zmqContext
        self.benchConfig = benchConfig
        self.modelSpec = modelSpec
        self.metrics = infbench.profCollection()
        self.nQuery = 0

        threading.Thread.__init__(self)

    def runBatch(self, queryBatch):
        for q in queryBatch:
            self.nQuery += 1
            inp = self.loader.get(q.index)
            sendReq(self.sutSock, q.id.to_bytes(8, sys.byteorder), inp)

    def processLatencies(self, latencies):
        self.metrics['t_response'] = infbench.processLatencies(self.benchConfig, latencies)

    def run(self):
        gpuType = infbench.getGpuType()

        self.loader = self.modelSpec.loader(self.modelSpec.dataDir)

        self.sutSock = self.zmqContext.socket(zmq.PAIR)
        self.sutSock.connect(sutSockUrl)

        runSettings = self.modelSpec.modelClass.getMlPerfCfg(gpuType, self.benchConfig)

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
        sendReq(self.serverSocket, self.benchConfig['model'].encode('utf-8'), b'')

        # PreWarm
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

    metrics = infbench.profCollection()
    metrics.merge(testRunner.metrics)
    metrics.merge(mlPerfMetrics)

    infbench.saveReport(metrics.report(), None, benchConfig, benchConfig['name'] + '_results.json')
