import numpy as np
import pathlib
import tempfile
import os
import sys
import io

# Defaults to home dir which I don't want. Have to set the env before loading
# the module because of python weirdness.
if "TEST_DATA_ROOT_PATH" not in os.environ:
    os.environ['TEST_DATA_ROOT_PATH'] = os.path.join(os.getcwd(), "downloads")

import onnx
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib import graph_executor
from tvm.contrib.download import download_testdata


def onnxPathToLibrary(onnxPath):
    """Convert an onnx path to a path to the cached shared library (this
    manipulates paths only, the library may or may not already exist."""
    libraryPath = pathlib.Path.cwd() / "modelCache" / (onnxPath.stem + ".so")
    if not libraryPath.parent.exists():
        libraryPath.parent.mkdir(mode=0o700)
    return libraryPath


def loadModelBuffer(onnxPath, create=False):
    """Return a model buffer from a pre-compiled .so."""
    libraryPath = onnxPathToLibrary(onnxPath)

    if not libraryPath.exists():
        raise RuntimeError("Pre-compiled model does not exist at " + str(libraryPath))

    with open(libraryPath, "rb") as f:
        model = f.read()
    return model


def importModelBuffer(buf):
    """Load the graph executor from an in-memory buffer containing the .so 
    file. This is a bit roundabout (we have to write it to a tmp file anyway),
    but it may be useful in faas settings."""
    with tempfile.TemporaryDirectory() as dpath:
        fpath = os.path.join(dpath, 'tmp.so')
        with open(fpath, 'wb') as f:
            f.write(buf)

        graphMod = tvm.runtime.load_module(fpath)

    ex = graph_executor.GraphModule(graphMod['default'](tvm.cuda()))
    return ex


def importOnnx(onnxPath, shape):
    """Will load the onnx model (*.onnx) from onnxPath and store it in a
    pre-compiled .so file. It will return a TVM executor capable of running the
    model. Shape is the input shape for the model and must be derived
    manually."""
    libraryPath = onnxPathToLibrary(onnxPath)

    if libraryPath.exists():
        graphMod = tvm.runtime.load_module(libraryPath)
    else:
        # This seems to be a convention in ONNX. There doesn't seem to be a
        # principled way to discover the input name in general.
        shapeDict = {"1" : shape}
        target = tvm.target.cuda()

        onnxModel = onnx.load(onnxPath)

        mod, params = relay.frontend.from_onnx(onnxModel, shapeDict)
        with tvm.transform.PassContext(opt_level=1):
            graphMod = relay.build(mod, target, params=params)

        graphMod.export_library(libraryPath)

    return graph_executor.GraphModule(graphMod['default'](tvm.cuda()))



class tvmModelBase():
    """A generic onnx on tvm model. Derived models should set constants for:
        nOutput:  Number of outputs returned 
        inpMap:   tuple representing which outputs of dataset.pre to use. For
                  now, this may only have one value.
        inShape:  dimensions of the input arrays
        outShape: dimensions of output array
    """
    def __init__(self, onnx):
        if isinstance(onnx, pathlib.Path) or isinstance(onnx, str):
            self.ex = importOnnx(onnx, self.inShape)
        elif isinstance(onnx, bytes):
            self.ex = importModelBuffer(onnx)


    def _run(self, tvmArr):
        # self.ex.set_input('1', tvm.nd.array(dat))
        self.ex.set_input('1', tvmArr)
        self.ex.run()
        return self.ex.get_output(0, tvm.nd.empty(self.outShape)).asnumpy().tobytes()


class superRes(tvmModelBase):
    nOutput = 1
    inpMap = (1,)
    inShape = (1, 1, 224, 224) 
    outShape = (1, 1, 672, 672)


    def run(self, dat):
        datNp = np.frombuffer(dat, dtype=np.float32)
        datNp.shape = self.inShape
        return super()._run(tvm.nd.array(datNp))
