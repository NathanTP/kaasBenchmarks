import numpy as np
import pathlib
import tempfile
import os
import io
import abc
import json
import pickle
import collections
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import mxnet

# Gluoncv throws out some stupid warning about having both mxnet and torch,
# have to go through this nonsense to suppress it.
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import gluoncv.data

import mlperf_loadgen

# Defaults to home dir which I don't want. Have to set the env before loading
# the module because of python weirdness.
if "TEST_DATA_ROOT_PATH" not in os.environ:
    os.environ['TEST_DATA_ROOT_PATH'] = os.path.join(os.getcwd(), "downloads")

import onnx
import tvm
import tvm.relay as relay
from tvm.contrib import graph_executor

from . import bert


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

    info = {}

    # I assume the model has only one input from the host (the first one). I
    # suppose we could validate this somehow by parsing the graph but we're
    # just gonna assume for now.
    ins = []
    for inNode in onnxModel.graph.input:
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


def _loadSo(libraryPath):
    metaPath = libraryPath.with_suffix(".json")
    module = tvm.runtime.load_module(libraryPath)
    with open(metaPath, 'r') as f:
        meta = json.load(f)

    model = graph_executor.GraphModule(module['default'](tvm.cuda()))
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


class tvmModel(abc.ABC):
    """A generic onnx on tvm model. Concrete models must additionally provide
        - preMap, runMap, postMap: inputMap objects for thepre, run, and post
        functions. Each function input is a list in order [const,inp,pre,run].
    """
    def __init__(self, modelDesc):
        """See loadModel() for allowable values of modelDesc"""
        self.model, self.meta = loadModel(modelDesc)

    def run(self, dat):
        """Run the model against input 'dat'. Dat is expected to be a bytes
       object that can be converted to numpy/tvm and passed to the model as
       input."""

        # XXX Assuming only one input for run
        dat = dat[0]

        datNp = np.frombuffer(dat, dtype=self.meta['inType'])
        datNp.shape = self.meta['inShape']

        self.model.set_input(self.meta['inName'], tvm.nd.array(datNp))
        self.model.run()

        outputs = []
        for i, outMeta in enumerate(self.meta['outputs']):
            outputs.append(self.model.get_output(i).numpy().tobytes())

        return outputs

    @staticmethod
    def getConstants(modelDir):
        return None

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


# =============================================================================
# Individual Models
# =============================================================================
class superRes(tvmModel):
    preMap = inputMap(inp=(0,))
    runMap = inputMap(pre=(1,))
    postMap = inputMap(pre=(0,), run=(0,))
    nOutPre = 2
    nOutRun = 1
    nOutPost = 1

    noPost = False

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

        imgPil = Image.frombytes("YCbCr", (224, 224), imgPilRaw)

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

        return (pngBuf,)

    @staticmethod
    def getMlPerfCfg(testing=False):
        """Return a configuration for mlperf_inference. If testing==True, run a
        potentially invalid configuration that will run fast. This should ease
        testing for correctness."""
        settings = getDefaultMlPerfCfg()

        if testing:
            # MLperf detects an unatainable SLO pretty fast
            settings.server_target_qps = 3
            settings.server_target_latency_ns = 1000
        else:
            # Set this to the lowest qps that any system should be able to get
            # (benchmarks might fiddle with it to get a real measurement).
            settings.server_target_qps = 3

            # This is arbitrary for superRes
            settings.server_target_latency_ns = 1000000000

        return settings


def _centerCrop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img


def _resizeWithAspectratio(img, out_height, out_width, scale=87.5, inter_pol=cv2.INTER_LINEAR):
    height, width, _ = img.shape
    new_height = int(100. * out_height / scale)
    new_width = int(100. * out_width / scale)
    if height > width:
        w = new_width
        h = int(new_height * height / width)
    else:
        h = new_height
        w = int(new_width * width / height)
    img = cv2.resize(img, (w, h), interpolation=inter_pol)
    return img


class resnet50(tvmModel):
    noPost = True
    preMap = inputMap(inp=(0,))
    runMap = inputMap(pre=(0,))
    postMap = inputMap(run=(0,))
    nOutRun = 2
    nOutPre = 1
    nOutPost = nOutRun

    @staticmethod
    def pre(imgBuf):
        imgBuf = imgBuf[0]
        img = cv2.imdecode(np.frombuffer(imgBuf, dtype=np.uint8), flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        output_height, output_width, _ = [224, 224, 3]

        cv2_interpol = cv2.INTER_AREA
        img = _resizeWithAspectratio(img, output_height, output_width, inter_pol=cv2_interpol)
        img = _centerCrop(img, output_height, output_width)
        img = np.asarray(img, dtype='float32')

        # normalize image
        means = np.array([123.68, 116.78, 103.94], dtype=np.float32)
        img -= means

        img = img.transpose([2, 0, 1])

        return (img.tobytes(),)

    @staticmethod
    def post(label):
        raise AttributeError("resnet50 has no post-processing")

    @staticmethod
    def getMlPerfCfg(testing=False):
        settings = getDefaultMlPerfCfg()

        settings.server_target_qps = 3
        # if testing:
        #     settings.server_target_latency_ns = 1000
        # else:
        #     settings.server_target_latency_ns = 50000000

        return settings


cocoClassList = [u'person', u'bicycle', u'car', u'motorcycle', u'airplane',
                 u'bus', u'train', u'truck', u'boat', u'traffic light', u'fire hydrant',
                 u'stop sign', u'parking meter', u'bench', u'bird',
                 u'cat', u'dog', u'horse', u'sheep', u'cow', u'elephant',
                 u'bear', u'zebra', u'giraffe', u'backpack', u'umbrella',
                 u'handbag', u'tie', u'suitcase', u'frisbee', u'skis',
                 u'snowboard', u'sports ball', u'kite', u'baseball bat',
                 u'baseball glove', u'skateboard', u'surfboard', u'tennis racket',
                 u'bottle', u'wine', u'glass', u'cup', u'fork',
                 u'knife', u'spoon', u'bowl', u'banana', u'apple', u'sandwich',
                 u'orange', u'broccoli', u'carrot', u'hot dog', u'pizza',
                 u'donut', u'cake', u'chair', u'couch', u'potted plant',
                 u'bed', u'dining table', u'toilet', u'tv', u'laptop',
                 u'mouse', u'remote', u'keyboard', u'cell phone', u'microwave',
                 u'oven', u'toaster', u'sink', u'refrigerator', u'book',
                 u'clock', u'vase', u'scissors', u'teddy bear', u'hair drier',
                 u'toothbrush']


class ssdMobilenet(tvmModel):
    noPost = False
    preMap = inputMap(inp=(0,))
    runMap = inputMap(pre=(0,))
    postMap = inputMap(pre=(1,), run=(0, 1, 2))
    nOutPre = 2
    nOutRun = 3
    nOutPost = nOutRun

    @staticmethod
    def pre(imgBuf):
        imgBuf = imgBuf[0]
        imgRaw = cv2.imdecode(np.frombuffer(imgBuf, dtype=np.uint8), flags=cv2.IMREAD_COLOR)
        imgRaw = cv2.cvtColor(imgRaw, cv2.COLOR_BGR2RGB)

        imgRaw = cv2.resize(imgRaw, (512, 512), interpolation=cv2.INTER_LINEAR)

        imgRaw = mxnet.nd.array(imgRaw).astype('uint8')
        imgMod, imgOrig = gluoncv.data.transforms.presets.ssd.transform_test(imgRaw, short=512)

        return (imgMod.asnumpy().tobytes(), imgOrig.tobytes())

    @staticmethod
    def post(modelOuts):
        imgOrig = np.frombuffer(modelOuts[0], dtype=np.uint8)
        cIDs = np.frombuffer(modelOuts[1], dtype=np.float32)
        scores = np.frombuffer(modelOuts[2], dtype=np.float32)
        bboxes = np.frombuffer(modelOuts[3], dtype=np.float32)
        imgOrig.shape = (512, 512, 3)
        cIDs.shape = (1, 100, 1)
        scores.shape = (1, 100, 1)
        bboxes.shape = (1, 100, 4)

        gluoncv.utils.viz.plot_bbox(
            imgOrig,
            bboxes[0],
            scores[0],
            cIDs[0],
            class_names=cocoClassList,
        )

        # Can't figure out how to save to buffer, easier to just trick pyplot
        with io.BytesIO() as f:
            plt.savefig(f, format="png")
            pngBuf = f.getvalue()

        return pngBuf

    @staticmethod
    def getMlPerfCfg(testing=False):
        settings = getDefaultMlPerfCfg()

        # XXX No idea right now
        if testing:
            settings.server_target_latency_ns = 1000
        else:
            settings.server_target_latency_ns = 1000000000

        return settings


class bertModel(tvmModel):
    noPost = False
    preMap = inputMap(const=(0,), inp=(0,))
    runMap = inputMap(pre=(0, 1, 2))
    postMap = inputMap(inp=(0,), pre=(3,), run=(0, 1))
    nOutPre = 4
    nOutRun = 2
    nOutPost = 1

    @staticmethod
    def getConstants(modelDir):
        with open(modelDir / 'vocab.txt', 'rb') as f:
            vocab = f.read()
        return [vocab]

    @staticmethod
    def pre(inputs):
        vocab = inputs[0]
        example = inputs[1]

        # featurize() can handle batches, but we only support batch size 1 right
        # now
        inputIds, inputMask, segmentIds, otherFeature = bert.featurize([example], vocab)[0]
        inputIds = np.array(inputIds).astype(np.int64)[np.newaxis, :].tobytes()
        inputMask = np.array(inputMask).astype(np.int64)[np.newaxis, :].tobytes()
        segmentIds = np.array(segmentIds).astype(np.int64)[np.newaxis, :].tobytes()
        return (inputIds, inputMask, segmentIds, otherFeature)

    @staticmethod
    def post(inputs):
        example = inputs[0]
        feature = inputs[1]
        startLogits = inputs[2]
        endLogits = inputs[3]

        startLogits = np.frombuffer(startLogits, dtype=np.float32).tolist()
        endLogits = np.frombuffer(endLogits, dtype=np.float32).tolist()

        pred = bert.interpret(startLogits, endLogits, example, feature)
        return pred

    @staticmethod
    def getMlPerfCfg(testing=False):
        settings = getDefaultMlPerfCfg()

        # XXX No idea right now
        if testing:
            settings.server_target_latency_ns = 1000
        else:
            settings.server_target_latency_ns = 1000000000

        return settings


# =============================================================================
# MLPERF INFERENCE STUFF
# =============================================================================

# Use as nquery argument to mlperf_loadgen.ConstructQSL
# I'm not 100% sure what this does...
mlperfNquery = 32


def getDefaultMlPerfCfg(testing=False):
    settings = mlperf_loadgen.TestSettings()
    settings.scenario = mlperf_loadgen.TestScenario.Server

    if testing:
        settings.mode = mlperf_loadgen.TestMode.PerformanceOnly
    else:
        settings.mode = mlperf_loadgen.TestMode.FindPeakPerformance

    # Default is 99, keeping it here due to Ray's awful tail
    settings.server_target_latency_percentile = 0.9

    return settings


# I don't actually know what this is supposed to do, none of the examples
# I've seen actually use it. MLPerf needs a callback for it though.
def flushQueries():
    pass


def processLatencies(latencies):
    """Callback for mlperf to report results"""
    # latencies is a list of latencies for each query issued (in ns).
    print("nQuery: ", len(latencies))
    print("Total Time: ", sum(latencies) / 1E9)


def reportMlPerf():
    with open("mlperf_log_summary.txt", 'r') as f:
        fullRes = f.readlines()
        print("".join(fullRes[:28]))
