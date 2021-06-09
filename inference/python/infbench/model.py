import numpy as np
import pathlib
import tempfile
import os
import sys
import io
import abc
import json
from PIL import Image
import matplotlib.pyplot as plt

import mlperf_loadgen

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


def getCachePath(name):
    """Convert an onnx path to a path to the cached shared library (this
    manipulates paths only, the library may or may not already exist."""

    # The compiled TVM output
    libraryPath = pathlib.Path.cwd() / "modelCache" / (name + ".so")

    # Metadata about model that's normally encoded in the onnx protobuf
    metaPath = pathlib.Path.cwd() / "modelCache" / (name + ".json")

    if not libraryPath.parent.exists():
        libraryPath.parent.mkdir(mode=0o700)

    return libraryPath, metaPath


# These map data_type and elem_type fields from the onnx protobuf structure to numpy types.
# I don't know of a nice way to get this, I manually extracted this from the onnx protobuf definition:
# https://github.com/onnx/onnx/blob/master/onnx/onnx.in.proto#L483-L485
onnxTypes = {
        1  : "float32",
        2  : "uint8",
        3  : "int8",
        4  : "uint16",
        5  : "int16",
        6  : "int32",
        7  : "int64",
        8  : "string",
        9  : "bool",
        10 : "float16",
        11 : "float64",
        12 : "uint32",
        13 : "uint64",
        # 14 and 15 are complex numbers, hopefully we don't need those
}


# This function was written with a bit of trial and error. the onnxModel
# returned by onnx.load is a protobuf structure, you can print it and inspect
# different parts of it.
def getOnnxInfo(onnxDesc):
    """Get metadata from an onnx file or loaded protobuf object. Metadata
       returned includes at least:
        - inName/outName   : Name of the input/output node
        - inType/outType   : numpy type of the input/output
        - inShape/outShape : numpy compatible shape tuple for input/output
    """

    if isinstance(onnxDesc, str) or isinstance(onnxDesc, pathlib.Path):
        onnxModel = onnx.load(onnxDesc)
    else:
        onnxModel = onnxDesc

    info = {}

    # I assume the model has only one input from the host (the first one). I
    # suppose we could validate this somehow by parsing the graph but we're
    # just gonna assume for now.
    inNode = onnxModel.graph.input[0]
    info['inName'] = inNode.name
    info['inType'] = onnxTypes[inNode.type.tensor_type.elem_type]

    info['inShape'] = []
    for dim in inNode.type.tensor_type.shape.dim:
        info['inShape'].append(dim.dim_value)
    
    if len(onnxModel.graph.output) > 1:
        raise RuntimeError("ONNX Model has multiple outputs, we can't handle this yet")

    outNode = onnxModel.graph.output[0]
    info['outName'] = outNode.name
    info['outType'] = onnxTypes[outNode.type.tensor_type.elem_type]
    info['outShape'] = []
    for dim in outNode.type.tensor_type.shape.dim:
        info['outShape'].append(dim.dim_value)

    return info


def readModelBuf(libraryPath):
    if libraryPath.suffix == ".onnx":
        libraryPath, metaPath = getCachePath(libraryPath.stem)
    else:
        metaPath = libraryPath.with_suffix(".json")

    with open(libraryPath, 'rb') as f:
        modelBuf = f.read()
    with open(metaPath, 'r') as f:
        meta = json.load(f)

    return modelBuf, meta


def _loadSo(libraryPath):
    metaPath = libraryPath.with_suffix(".json")
    module = tvm.runtime.load_module(libraryPath)
    with open(metaPath, 'r') as f:
        meta = json.load(f)

    model = graph_executor.GraphModule(module['default'](tvm.cuda()))
    return model, meta


def _loadOnnx(onnxPath):
    onnxModel = onnx.load(onnxPath)
    meta = getOnnxInfo(onnxModel)

    mod, params = relay.frontend.from_onnx(onnxModel)
    with tvm.transform.PassContext(opt_level=1):
        module = relay.build(mod, tvm.target.cuda(), params=params)

    # Cache Results
    libraryPath, metaPath = getCachePath(onnxPath.stem)
    module.export_library(libraryPath)
    with open(metaPath, 'w') as f:
        json.dump(meta, f)

    model = graph_executor.GraphModule(module['default'](tvm.cuda()))
    return model, meta


def loadModel(modelDesc):
    """Load a saved model. modelDesc describes where to get the model, it can be either:
        - Path to onnx file: If modelDesc is a path to a .onnx file, loadModel
              will attempt to find a cached pre-compiled version of the onnx file
              first. If none is found, the onnx model will be compiled and cached.
        - Path to precompiled (.so) file: If modelDesc points to a .so,
              loadModel will assume this is a TVM pre-compiled library. It will
              look for a corresponding .json file with model metadata in the same
              directory as the .so and load from these. 
        - Tuple of (bytes, dict): modelDesc can be a pre-loaded raw model. In
              this case bytes is assumed to be the contents of a .so file and dict
              is the corresponding metadata.
    
    Paths are assumed to be pathlib.Path

    Returns:
        (TVM graph executor, onnxInfo dictionary)
        - see getOnnxInfo for details of the metadata dictionary.
    """
    if isinstance(modelDesc, pathlib.Path):
        if modelDesc.suffix == ".so":
            model, meta = _loadSo(modelDesc)

        elif modelDesc.suffix == ".onnx":
            libraryPath, metaPath = getCachePath(modelDesc.stem)
            if libraryPath.exists():
                model, meta = _loadSo(libraryPath)
            else:
                model, meta = _loadOnnx(modelDesc)

        else:
            raise RuntimeError("model description {} invalid (must be either an onnx file or a .so)".format(modelDesc))

    elif isinstance(modelDesc, tuple):
        modelBin = modelDesc[0]
        with tempfile.TemporaryDirectory() as dpath:
            fpath = os.path.join(dpath, 'tmp.so')
            with open(fpath, 'wb') as f:
                f.write(modelBin)

            graphMod = tvm.runtime.load_module(fpath)

        model = graph_executor.GraphModule(graphMod['default'](tvm.cuda()))
        meta = modelDesc[1]

    else:
        raise RuntimeError("Model description type {} invalid (must be .onnx, .so, or tuple)".format(type(modelDesc)))

    return model, meta


class tvmModel(abc.ABC):
    """A generic onnx on tvm model. Concrete models must additionally provide:
        - runMap:  Which output from pre() to pass to run. For now, there can
                   only be one input to run. 
        - postMap: List of indices to pass to post from pre()'s output. post()
                   will recieve these values, followed by the output of run(). 
    """
    def __init__(self, modelDesc):
        """See loadModel() for allowable values of modelDesc"""
        self.model, self.meta = loadModel(modelDesc)

    def run(self, dat):
        """Run the model against input 'dat'. Dat is expected to be a bytes
       object that can be converted to numpy/tvm and passed to the model as
       input."""
        datNp = np.frombuffer(dat, dtype=self.meta['inType'])
        datNp.shape = self.meta['inShape']

        self.model.set_input(self.meta['inName'], tvm.nd.array(datNp))
        self.model.run()
        return self.model.get_output(0, tvm.nd.empty(self.meta['outShape'])).asnumpy().tobytes()


    @staticmethod
    @abc.abstractmethod
    def pre(data):
        """Preprocess data and return nOutputPre bytes objects"""
        pass


    @staticmethod
    @abc.abstractmethod
    def post(data):
        """Postprocess data and return nOutputPost bytes objects"""
        pass


    @staticmethod
    @abc.abstractmethod
    def getMlPerfCfg(testing=False):
        """Return a mlperf settings object for this model"""
        pass


class superRes(tvmModel):
    postMap = (0,)
    runMap = 1
    nOutPre = 2
    nOutPost = 1

    @staticmethod
    def pre(data):
        raw = data[0]
        # mode and size were manually read from the png (used Image.open and then
        # inspected the img.mode and img.size attributes). We're gonna just go with
        # this for now.
        img = Image.frombytes("RGB", (256,256), raw)

        imgProcessed = img.resize((224,224)).convert("YCbCr")
        img_y, img_cb, img_cr = imgProcessed.split()
        imgNp = (np.array(img_y)[np.newaxis, np.newaxis, :, :]).astype("float32")

        return (imgProcessed.tobytes(), imgNp.tobytes())

    @staticmethod
    def post(data):
        imgPilRaw = data[0]
        imgRetRaw = data[1]

        retNp = np.frombuffer(imgRetRaw, dtype=np.float32)
        retNp.shape = (1, 1, 672, 672)
        retNp = np.uint8((retNp[0, 0]).clip(0, 255))

        imgPil = Image.frombytes("YCbCr", (224,224), imgPilRaw)

        img_y, img_cb, img_cr = imgPil.split()
        out_y = Image.fromarray(retNp, mode="L")
        out_cb = img_cb.resize(out_y.size, Image.BICUBIC)
        out_cr = img_cr.resize(out_y.size, Image.BICUBIC)
        result = Image.merge("YCbCr", [out_y, out_cb, out_cr]).convert("RGB")
        canvas = np.full((672, 672 * 2, 3), 255)
        canvas[0:224, 0:224, :] = np.asarray(imgPil)
        canvas[:, 672:, :] = np.asarray(result)

        with io.BytesIO() as f:
            plt.imsave(f, canvas.astype(np.uint8), format="png")
            pngBuf = f.getvalue()

        return (pngBuf)


    @staticmethod
    def getMlPerfCfg(testing=False):
        """Return a configuration for mlperf_inference. If testing==True, run a
        potentially invalid configuration that will run fast. This should ease
        testing for correctness."""
        settings = getDefaultMlPerfCfg()

        if testing:
            # MLperf detects an unatainable SLO pretty fast.
            settings.server_target_qps = 3
            settings.server_target_latency_ns = 10000000
            settings.min_duration_ms = 1000
        else:
            # Set this to the lowest qps that any system should be able to get
            # (benchmarks might fiddle with it to get a real measurement).
            settings.server_target_qps = 3

            # This is arbitrary for superRes
            settings.server_target_latency_ns = 1000000000

        return settings


#==============================================================================
# MLPERF INFERENCE STUFF
#==============================================================================

# Use as nquery argument to mlperf_loadgen.ConstructQSL
# I'm not 100% sure what this does... 
mlperfNquery = 32


def getDefaultMlPerfCfg():
    settings = mlperf_loadgen.TestSettings()
    settings.scenario = mlperf_loadgen.TestScenario.Server
    settings.mode = mlperf_loadgen.TestMode.FindPeakPerformance

    # I don't think these are all that big of a deal, but you can bump them
    # up if results aren't statistically significant enough.
    settings.min_query_count = 10
    # settings.min_query_count = 200
    settings.min_duration_ms = 10000

    return settings


# I don't actually know what this is supposed to do, none of the examples
# I've seen actually use it. MLPerf needs a callback for it though.
def flushQueries():
    pass


def processLatencies(latencies):
    """Callback for mlperf to report results"""
    # latencies is a list of latencies for each query issued (in ns).
    # For now we leave this blank because the benchmarks report the final
    # results from mlperf's logs. This could be useful for custom analysis or
    # something, but we don't need it now.
    pass

def reportMlPerf():
    with open("mlperf_log_summary.txt", 'r') as f:
        fullRes = f.readlines()
        print("".join(fullRes[:28]))
