Example of running onnx models in TVM. This is the basic flow for an eventual
inference benchmark. This one is hard-coded to run a basic super resolution
model. It was adapted from [this tvm tutorial](https://tvm.apache.org/docs/tutorials/frontend/from_onnx.html).

# Installation and Usage
You will need a recent version of TVM for this to work. It was tested on TVMs
main branch as of May 27 2021.

The TVM-only example:

    python serve.py

To use a Ray example, you'll need to run the tvm-only example first to generate
the TVM shared library and sample data. Then you can run it as:

    python serveRay.py
