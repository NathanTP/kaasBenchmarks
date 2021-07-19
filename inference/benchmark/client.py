import sys
import argparse
import util
import zmq
import pickle


def setupZmq(clientID):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="testModelNP", help="Model name")
    parser.add_argument("-n", "--name", type=str, default="client0", help="Unique identifier for this client")

    args = parser.parse_args()

    clientID = args.name.encode("utf-8")
    nShot(args.model, clientID, 1)
    print("Test Complete")


if __name__ == "__main__":
    main()
