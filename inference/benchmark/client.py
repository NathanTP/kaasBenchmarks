#!/usr/bin/env python

import sys
import pickle
import threading
from tornado.ioloop import IOLoop
import zmq
from zmq.eventloop.zmqstream import ZMQStream
from pprint import pprint
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


def preWarm(serverSock, barrierSock, inputs):
    nReq = util.getNGpu()*2
    for i in range(nReq):
        serverSock.send_multipart([bytes(1), pickle.dumps(inputs)])
    for i in range(nReq):
        serverSock.recv_multipart()

    barrier(barrierSock)


def nShot(modelSpec, n, benchConfig):
    """A simple nShot test. All N requests are sent before waiting for
    responses. The raw results are returned."""
    clientID = benchConfig['name'].encode('utf-8')

    stats = util.profCollection()

    loader = modelSpec.loader(modelSpec.dataDir)
    loader.preLoad(range(min(n, loader.ndata)))

    zmqContext = zmq.Context()
    serverSocket, barrierSocket = setupZmq(clientID, context=zmqContext)

    # Registration
    print("Registering client: ", benchConfig['name'])
    serverSocket.send_multipart([benchConfig['model'].encode('utf-8'), b''])

    print("Waiting for other clients:")
    barrier(barrierSocket)

    # Send all requests
    print("Sending Requests:")
    results = []
    accuracies = []
    for i in range(n):
        idx = i % loader.ndata
        inp = loader.get(idx)

        with util.timer('t_e2e', stats):
            serverSocket.send_multipart([idx.to_bytes(4, sys.byteorder), pickle.dumps(inp)])

            respIdx, respData = serverSocket.recv_multipart()
            respIdx = int.from_bytes(respIdx, sys.byteorder)

            assert (respIdx == idx)
            results.append(pickle.loads(respData))

        if loader.checkAvailable:
            accuracies.append(loader.check(results[-1], idx))

    # Check Accuracy
    print("Test complete, checking accuracies")
    if loader.checkAvailable:
        print("Accuracy = ", sum([int(res) for res in accuracies]) / n)

    report = stats.report()
    print("E2E Results:")
    pprint({(k, v) for (k, v) in report['t_e2e'].items() if k != "events"})

    return results


class mlperfRunner(threading.Thread):
    def __init__(self, modelSpec, benchConfig, zmqContext):
        self.zmqContext = zmqContext
        self.benchConfig = benchConfig
        self.modelSpec = modelSpec
        self.metrics = {}
        self.nQuery = 0

        threading.Thread.__init__(self)

    def runBatch(self, queryBatch):
        for q in queryBatch:
            self.nQuery += 1
            inp = self.loader.get(q.index)
            self.sutSock.send_multipart([q.id.to_bytes(8, sys.byteorder), pickle.dumps(inp)])

    def processLatencies(self, latencies):
        self.metrics = {**infbench.model.processLatencies(self.benchConfig, latencies), **self.metrics}

    def run(self):
        gpuType = util.getGpuType()

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
        self.metrics['t_e2e'] = time.time() - start

        mlperf_loadgen.DestroyQSL(qsl)
        mlperf_loadgen.DestroySUT(sut)

        # Shutdown Signal
        self.sutSock.send_multipart([b'', b''])
        self.sutSock.close()

        self.metrics['n_query'] = self.nQuery


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
        self.serverSocket.send_multipart([self.benchConfig['model'].encode('utf-8'), b''])

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
        reqID, data = msg

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
    mlPerfMetrics = infbench.model.parseMlPerf(benchConfig['name'] + '_')
    infbench.model.saveReport({**testRunner.metrics, **mlPerfMetrics},
                              benchConfig, benchConfig['name'] + '_results.json')
