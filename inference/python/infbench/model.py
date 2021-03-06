import numpy as np
import pathlib
import tempfile
import os
import sys
import abc
import json
import yaml
import pickle
import collections
import ray

import mlperf_loadgen

import kaas
import kaas.ray

# Defaults to home dir which I don't want. Have to set the env before loading
# the module because of python weirdness.
if "TEST_DATA_ROOT_PATH" not in os.environ:
    os.environ['TEST_DATA_ROOT_PATH'] = os.path.join(os.getcwd(), "downloads")

import onnx
import tvm
import tvm.relay as relay
from tvm.contrib import graph_executor


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
    1:  "float32",
    2:  "uint8",
    3:  "int8",
    4:  "uint16",
    5:  "int16",
    6:  "int32",
    7:  "int64",
    8:  "string",
    9:  "bool",
    10: "float16",
    11: "float64",
    12: "uint32",
    13: "uint64",
    # 14 and 15 are complex numbers, hopefully we don't need those
}


# This function was written with a bit of trial and error. the onnxModel
# returned by onnx.load is a protobuf structure, you can print it and inspect
# different parts of it.
def getOnnxInfo(onnxDesc):
    """Get metadata from an onnx file or loaded protobuf object. Metadata
       returned includes at least:
        - inName   : Name of the input/output node
        - inType   : numpy type of the input/output
        - inShape  : numpy compatible shape tuple for input/output
        - outputs : list of dicts of name, type, shape  for outputs (same keys/meanings as for in*)
    """

    if isinstance(onnxDesc, str) or isinstance(onnxDesc, pathlib.Path):
        onnxModel = onnx.load(onnxDesc)
    else:
        onnxModel = onnxDesc

    initializers = []
    for node in onnxModel.graph.initializer:
        initializers.append(node.name)

    info = {}

    # I assume the model has only one input from the host (the first one). I
    # suppose we could validate this somehow by parsing the graph but we're
    # just gonna assume for now.
    ins = []
    for inNode in onnxModel.graph.input:
        # Some onnx graphs (like superRes) have explicit inputs for
        # initializers (weights). We only want to record dynamic inputs. Most
        # models don't do this.
        if inNode.name in initializers:
            continue

        inInfo = {}
        inInfo['name'] = inNode.name
        inInfo['type'] = onnxTypes[inNode.type.tensor_type.elem_type]

        inInfo['shape'] = []
        for dim in inNode.type.tensor_type.shape.dim:
            inInfo['shape'].append(dim.dim_value)
        ins.append(inInfo)
    info['inputs'] = ins

    outs = []
    for outNode in onnxModel.graph.output:
        outInfo = {'name': outNode.name,
                   'type': onnxTypes[outNode.type.tensor_type.elem_type]}

        outInfo['shape'] = []
        for dim in outNode.type.tensor_type.shape.dim:
            outInfo['shape'].append(dim.dim_value)
        outs.append(outInfo)

    info['outputs'] = outs

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


def _loadSo(libraryPath, devID=None):
    metaPath = libraryPath.with_suffix(".json")
    module = tvm.runtime.load_module(libraryPath)
    with open(metaPath, 'r') as f:
        meta = json.load(f)

    if devID is None:
        model = graph_executor.GraphModule(module['default'](tvm.cuda()))
    else:
        model = graph_executor.GraphModule(module['default'](tvm.cuda(dev_id=devID)))

    return model, meta


def _loadOnnx(onnxPath, cache=True):
    onnxModel = onnx.load(onnxPath)
    meta = getOnnxInfo(onnxModel)

    # Some onnx models seem to have dynamic parameters. I don't really know
    # what this is but the freeze_params and DynamicToStatic call seem to
    # resolve the issue.
    mod, params = relay.frontend.from_onnx(onnxModel, freeze_params=True)
    relay.transform.DynamicToStatic()(mod)
    with tvm.transform.PassContext(opt_level=3):
        module = relay.build(mod, tvm.target.cuda(), params=params)

    # Cache Results
    if cache:
        libraryPath, metaPath = getCachePath(onnxPath.stem)
        module.export_library(libraryPath)
        with open(metaPath, 'w') as f:
            json.dump(meta, f)

    model = graph_executor.GraphModule(module['default'](tvm.cuda()))
    return model, meta


def loadModel(modelDesc, devID=None):
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

    devID: Optional explicit device ID to use instead of CUDA_VISIBLE_DEVICES
    Paths are assumed to be pathlib.Path

    Returns:
        (TVM graph executor, onnxInfo dictionary)
        - see getOnnxInfo for details of the metadata dictionary.
    """
    if isinstance(modelDesc, pathlib.Path):
        if modelDesc.suffix == ".so":
            model, meta = _loadSo(modelDesc, devID=devID)

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

        if devID is None:
            model = graph_executor.GraphModule(graphMod['default'](tvm.cuda()))
        else:
            model = graph_executor.GraphModule(graphMod['default'](tvm.cuda(dev_id=devID)))

        meta = modelDesc[1]

    else:
        raise RuntimeError("Model description type {} invalid (must be .onnx, .so, or tuple)".format(type(modelDesc)))

    return model, meta


def dumpModel(graphMod, outputBasePath):
    lib = graphMod.get_lib()
    cudaLib = lib.imported_modules[0]

    print("\n")
    print(graphMod.__dir__())

    graphPath = outputBasePath.with_suffix(".graph.json")
    print("Saving execution graph to: ", graphPath)
    with open(graphPath, 'w') as f:
        f.write(graphMod.get_graph_json())

    paramPath = outputBasePath.with_suffix(".params.pickle")
    print("Saving parameters to: ", paramPath)
    with open(paramPath, "wb") as f:
        # Can't pickle tvm.ndarray, gotta convert to numpy
        pickle.dump({k: p.asnumpy() for k, p in graphMod.params.items()}, f)

    srcPath = outputBasePath.with_suffix(".cu")
    print("Saving Raw CUDA Source to: ", srcPath)
    with open(srcPath, 'w') as f:
        f.write(cudaLib.get_source())

    print("Saving CUDA to: {}.ptx and {}.tvm_meta.json:".format(outputBasePath.stem, outputBasePath.stem))
    cudaLib.save(str(outputBasePath.with_suffix(".ptx")))


# Each field of the input map is a list of indices from that step that should
# be passed as input.
inputMap = collections.namedtuple("inputMap", ["const", "inp", "pre", "run"], defaults=[None]*4)


class Model(abc.ABC):
    """Base class for all models. The benchmark expects this interface."""

    def __init__(self, *args, **kwargs):
        pass

    @property
    @abc.abstractmethod
    def preMap(self) -> inputMap:
        """Input map for preprocessing"""
        ...

    @property
    @abc.abstractmethod
    def runMap(self) -> inputMap:
        """Input map for the model"""
        ...

    @property
    @abc.abstractmethod
    def postMap(self) -> inputMap:
        """Input map for postprocessing"""
        ...

    @property
    @abc.abstractmethod
    def nOutPre(self):
        """Number of outputs from preprocessing"""
        ...

    @property
    @abc.abstractmethod
    def nOutRun(self):
        """Number of outputs from the model"""
        ...

    @property
    @abc.abstractmethod
    def nOutPost(self):
        """Number of outputs from postprocessing"""
        ...

    noPre = False

    @property
    @abc.abstractmethod
    def noPost(self) -> bool:
        """Does the model support post-processing?"""
        ...

    @property
    @abc.abstractmethod
    def nConst(self):
        """Number of constants returned by getConstants"""
        ...

    @staticmethod
    def getConstants(modelDir):
        return None

    @staticmethod
    @abc.abstractmethod
    def pre(data):
        """Preprocess data and return nOutputPre bytes objects"""
        pass

    @abc.abstractmethod
    def run(self, dat, stats=None):
        """Run the model against input 'dat'. The format of dat is dependent on
        the concrete model type."""
        ...

    @staticmethod
    @abc.abstractmethod
    def post(data):
        """Postprocess data and return nOutputPost bytes objects"""
        pass

    def shutdown(self):
        """Implement this method if you'd like something to happen when the
        model is no longer needed"""
        pass


class tvmModel(Model):
    """A generic tvm model. Should be initialized with a precompiled .so"""

    def __init__(self, modelDesc, devID=None):
        self.model, self.meta = loadModel(modelDesc, devID=devID)

    def run(self, dat, stats=None):
        """Run the model against input 'dat'. Dat is expected to be a bytes
        object that can be converted to numpy/tvm and passed to the model as
        input."""
        for idx, inpMeta in enumerate(self.meta['inputs']):
            inputDat = dat[idx]
            datNp = np.frombuffer(inputDat, dtype=inpMeta['type'])
            datNp.shape = inpMeta['shape']
            self.model.set_input(inpMeta['name'], tvm.nd.array(datNp))

        self.model.run()
        # dev = tvm.device('cuda', 0)
        # ftimer = self.model.module.time_evaluator("run", dev)
        # prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        # print("Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res)))

        outputs = []
        for i, outMeta in enumerate(self.meta['outputs']):
            outputs.append(self.model.get_output(i).numpy().tobytes())

        return tuple(outputs)


class kaasModel(Model):
    """A generic KaaS model."""
    def __init__(self, modelArg, constRefs, backend='ray'):
        """Can be initialized either by an existing kaasModel or by a path to a
        KaaS model. If a path is passed, it should be a directory containing:
        name.cubin, name_meta.yaml, and name_model.yaml (where name is the
        name of the directory)."""

        self.backend = backend

        # In some cases, it's easier to pass a pre-initialized model as an
        # argument, typically to keep abstractions clean on the client side.
        if isinstance(modelArg, kaasModel):
            self.cubin = modelArg.cubin
            self.reqTemplate = modelArg.reqTemplate
            self.meta = modelArg.meta
        else:
            modelDir = modelArg.parent

            baseName = modelDir.stem
            self.cubin = modelDir / (baseName + ".cubin")

            with open(modelDir / (baseName + "_model" + ".pkl"), 'rb') as f:
                req = pickle.load(f)

            with open(modelDir / (baseName + "_meta" + ".yaml"), 'r') as f:
                self.meta = yaml.safe_load(f)

        for i in range(len(req.kernels)):
            kern = req.kernels[i]
            if kern.library is None:
                req.kernels[i] = kaas.denseKern(kern.name, self.cubin,
                                                kern.kernel, kern.gridDim,
                                                kern.blockDim, kern.sharedSize,
                                                kern.literals, kern.arguments,
                                                kern.ioTypes)

        renameMap = {}
        if self.backend == 'ray':
            for const in self.meta['constants']:
                renameMap[const['name']] = kaas.ray.putObj(constRefs[const['dataIdx']])
        else:
            for const in self.meta['constants']:
                renameMap[const['name']] = constRefs[const['dataIdx']]

        req.reKey(renameMap)

        if self.backend == 'ray':
            self.reqRef = ray.put(req)
        else:
            self.reqRef = req

    @staticmethod
    def getConstants(modelDir):
        """Default constant loader assumes the kaasModel simply pickled their
        constants and we can load them directly."""
        paramPath = modelDir / (modelDir.stem + "_params.pkl")
        if paramPath.exists():
            with open(paramPath, 'rb') as f:
                constants = pickle.load(f)
            return constants
        else:
            return []

    def run(self, dat, outKeys=None, stats=None):
        """Unlike other Models, kaas accepts keys or references to inputs in
        dat rather than actual values. Run here will submit the model to KaaS
        and returns a list of references/keys to the outputs."""
        inputs = dat[self.nConst:]

        renameMap = {}

        if self.backend == 'ray':
            for idx, inp in enumerate(self.meta['inputs']):
                renameMap[inp['name']] = kaas.ray.putObj(inputs[idx])
        else:
            for idx, inp in enumerate(self.meta['inputs']):
                renameMap[inp['name']] = inputs[idx]

        if outKeys is not None:
            outNames = [out['name'] for out in self.meta['outputs']]
            for name, key in zip(outNames, outKeys):
                renameMap[name] = key

        return (self.reqRef, renameMap)


# =============================================================================
# MLPERF INFERENCE STUFF
# =============================================================================

# Used by ConstructQSL: it's the number of samples to hold in memory. I believe
# the loadgen driver will cache these and otherwise re-query our loaders which
# all cache anyway.
mlperfNquery = 128


def getDefaultMlPerfCfg(maxQps, medianLat, benchConfig):
    """Return a configuration for the MlPerf benchmark based on the model and
    benchmark properties. All metrics are based on client/server mode. For
    agpu1 (K20c) we measure throughput for both GPUs, for EC2 (V100), we
    measure on one GPU only.
        - maxQps: Peak QPS achieved by mlperf FindPeakPerformance using the
        balance policy and actors/kaas. As a peak, you should generally run a
        bit lower in PerformanceOnly mode using the --scale option.

        - medianLat: Median latency recorded using nshot with 32 iterations
        using the balance policy and actors/kaas.
    """

    settings = mlperf_loadgen.TestSettings()
    settings.scenario = mlperf_loadgen.TestScenario.Server

    # Default is 99, 90 is a bit more manageable in terms of avoiding tricky
    # tuning and requiring very long tests
    # settings.server_target_latency_percentile = 0.9
    # settings.server_target_latency_ns = int((medianLat*4)*1E9)
    settings.server_target_latency_percentile = 0.99
    settings.server_target_latency_ns = int((medianLat*10)*1E9)

    settings.schedule_rng_seed = int.from_bytes(os.urandom(8), byteorder=sys.byteorder)

    if benchConfig['scale'] is None:
        # Even in client/server mode, if FindPeakPerformance starts with a QPS
        # close to max it will experience a severe outlier. I don't know why
        # this happens since we pre-warm, but it does. Start from a much lower
        # QPS and ramp up to peak for FindPeakPerformance mode.
        settings.server_target_qps = maxQps * 0.25
        settings.mode = mlperf_loadgen.TestMode.FindPeakPerformance
    else:
        settings.mode = mlperf_loadgen.TestMode.PerformanceOnly
        settings.server_target_qps = benchConfig['scale']

        if benchConfig['runTime'] is None:
            settings.min_duration_ms = int(300*1E3)
            settings.max_duration_ms = int(300*1E3)
        else:
            settings.min_duration_ms = int(benchConfig['runTime']*1E3)
            settings.max_duration_ms = int(benchConfig['runTime']*1E3)

    return settings


# I don't actually know what this is supposed to do, none of the examples
# I've seen actually use it. MLPerf needs a callback for it though.
def flushQueries():
    pass
