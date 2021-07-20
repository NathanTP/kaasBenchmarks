# First Time Setup

## Getting Models
You can use tools/getModels.py to fetch the various models we want to use. They
should total a few hundred MB. These will be stored in models/.

## Getting Data
Data is more complicated than the models, we are still working on an elegant
solution. For now, you can use the tools/make\_fake\_imagenet.sh to create a
simple dataset that is sufficient for testing. You can manually get other
datasets by following the instructions sprinkled throughout
https://github.com/mlcommons/inference

# Remote Client Protocol
The benchmark supports a server mode where multiple clients can connect and
request simultaneous workloads. This facilitates a multitenant workload with
each client running a copy of mlperf loadgen. Requests are sent over zmq
send\_multipart() using the following protocol:

## Register
Before sending requests, the client must register with the server. Registration
takes the following form:
* clientID: A unique clientID, this cannot change during the run. Clients are
  responsible for ensuring this is unique.
* modelName: A string describing which model/workload will be used. utf-8
  encoded string.
* b'': An empty frame (needed to keep register and request messages consistent)

*Implementation Note: If an unrecognized clientID is sent, the server assumes
this is a registration message, otherwise it assumes it's a request.* 

## Request
There are three zmq parts (send\_multipart):

* clientID: A unique clientID, this cannot change during the run.
* reqID: An arbitrary number identifying this query. It is used by the client
  to connect responses to requests. The server does not use this field. Most
  likely, this should be the queryID for mlperf loadgen.
* payload: Raw binary input to the model.

The response is as follows:

* reqID: Corresponds to the reqID provided with the request.
* result: Binary data representing the result of the computation.
