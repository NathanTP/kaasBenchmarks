import sys
import argparse
import pickle
import threading
from tornado.ioloop import IOLoop
import zmq
from zmq.eventloop.zmqstream import ZMQStream

import mlperf_loadgen

import infbench
import util

sutSockUrl = "inproc://sut"


def setupZmq(clientID, context=None):
    if context is None:
        context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    socket.identity = clientID
    socket.connect(util.clientUrl)
    return socket


def nShot(modelName, clientID, n):
    """A simple nShot test. All N requests are sent before waiting for
    responses. The raw results are returned."""

    modelSpec = util.getModelSpec(modelName)
    loader = modelSpec.loader(modelSpec.dataDir)
    loader.preLoad(range(min(n, loader.ndata)))

    socket = setupZmq(clientID)

    # Registration
    print("Registering client: ", clientID)
    socket.send_multipart([modelName.encode('utf-8'), b''])

    # Send all requests
    print("Sending Requests:")
    for i in range(n):
        idx = i % loader.ndata
        inp = loader.get(idx)

        socket.send_multipart([idx.to_bytes(4, sys.byteorder), pickle.dumps(inp)])

    # Wait for responses
    print("Waiting for responses")
    responses = {}  # {idx -> data}
    for i in range(n):
        idx, respData = socket.recv_multipart()
        responses[int.from_bytes(idx, sys.byteorder)] = pickle.loads(respData)

    # Check Accuracy
    print("Test complete, checking accuracies")
    if loader.checkAvailable:
        accuracies = []
        for idx, data in responses.items():
            accuracies.append(loader.check(data, idx))

        print("Accuracy = ", sum([int(res) for res in accuracies]) / n)

    return [data for idx, data in responses.items()]


class mlperfRunner(threading.Thread):
    def __init__(self, clientID, modelName, zmqContext, testing=False):
        self.zmqContext = zmqContext
        self.testing = testing
        self.modelName = modelName

        threading.Thread.__init__(self)

    def runBatch(self, queryBatch):
        for q in queryBatch:
            inp = self.loader.get(q.index)
            self.sutSock.send_multipart([q.id.to_bytes(8, sys.byteorder), pickle.dumps(inp)])

    def preWarm(self):
        self.loader.preLoad([0])
        inputs = self.loader.get(0)
        self.sutSock.send_multipart([bytes(1), pickle.dumps(inputs)])

    def run(self):
        modelSpec = util.getModelSpec(self.modelName)
        self.loader = modelSpec.loader(modelSpec.dataDir)

        self.sutSock = self.zmqContext.socket(zmq.PAIR)
        self.sutSock.connect(sutSockUrl)

        self.preWarm()

        settings = modelSpec.modelClass.getMlPerfCfg(testing=self.testing)
        sut = mlperf_loadgen.ConstructSUT(
            self.runBatch, infbench.model.flushQueries, infbench.model.processLatencies)

        qsl = mlperf_loadgen.ConstructQSL(
            self.loader.ndata, infbench.model.mlperfNquery, self.loader.preLoad, self.loader.unLoad)

        mlperf_loadgen.StartTest(sut, qsl, settings)
        mlperf_loadgen.DestroyQSL(qsl)
        mlperf_loadgen.DestroySUT(sut)

        # Shutdown Signal
        self.sutSock.send_multipart([b'', b''])
        self.sutSock.close()


class mlperfLoop():
    def __init__(self, clientID, modelName, zmqContext):
        self.loop = IOLoop.instance()

        self.sutSocket = zmqContext.socket(zmq.PAIR)
        self.sutSocket.bind(sutSockUrl)
        self.sutStream = ZMQStream(self.sutSocket)
        self.sutStream.on_recv(self.handleSut)

        self.serverSocket = setupZmq(clientID, context=zmqContext)
        self.serverStream = ZMQStream(self.serverSocket)
        self.serverStream.on_recv(self.handleServer)

        # Register with the benchmark server
        self.serverSocket.send_multipart([modelName.encode('utf-8'), b''])

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


def mlperfBench(modelName, clientID):
    context = zmq.Context()

    testRunner = mlperfRunner(clientID, modelName, context, testing=False)
    testRunner.start()

    mlperfLoop(clientID, modelName, context)
    IOLoop.instance().start()

    infbench.model.reportMlPerf()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="testModelNP", help="Model name")
    parser.add_argument("-n", "--name", type=str, default="client0", help="Unique identifier for this client")

    args = parser.parse_args()

    clientID = args.name.encode("utf-8")
    # nShot(args.model, clientID, 1)
    mlperfBench(args.model, clientID)


if __name__ == "__main__":
    main()
