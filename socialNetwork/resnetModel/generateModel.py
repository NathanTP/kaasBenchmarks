#!/usr/bin/env python3
import yaml
import pathlib
import pickle
import json
import wget
import subprocess as sp
import onnx
import tvm
import tvm.relay as relay

from resnet50 import createReq
#XXX
import infbench

modelDir = pathlib.Path.cwd()
outDir = (modelDir / 'resnet50').resolve()


# These map data_type and elem_type fields from the onnx protobuf structure to numpy types.
# I don't know of a nice way to get this, I manually extracted this from the onnx protobuf definition:
# https://github.com/onnx/onnx/blob/master/onnx/onnx.in.proto#L483-L485
onnxTypes = {
        1: "float32",
        2: "uint8",
        3: "int8",
        4: "uint16",
        5: "int16",
        6: "int32",
        7: "int64",
        8: "string",
        9: "bool",
        10: "float16",
        11: "float64",
        12: "uint32",
        13: "uint64",
        # 14 and 15 are complex numbers, hopefully we don't need those
}


def fixOnnxDim(model, inputMap):
    """Some onnx models have dynamic input shapes (usually for the batch size).
    This function sets the input shape to 'inputMap' and returns a static
    onnx model.
    inputMap format: { inputName : shape }"""
    for node in model.graph.input:
        name = node.name
        if name in inputMap:
            print('Changing input "{}" dimension to: {}'.format(name, inputMap[name]))
            for dim, new in zip(node.type.tensor_type.shape.dim, inputMap[name]):
                dim.dim_value = new
        else:
            for dim in node.type.tensor_type.shape.dim:
                if dim.dim_param != "":
                    print("WARNING: input {} has dynamic dimension but was not replaced".format(name))

    return model


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


# This function was written with a bit of trial and error. the onnxModel
# returned by onnx.load is a protobuf structure, you can print it and inspect
# different parts of it.
# def getOnnxInfo(onnxModel):
#     """Get metadata from an onnx object. Metadata returned includes at least:
#         - inName/outName   : Name of the input/output node
#         - inType/outType   : numpy type of the input/output
#         - inShape/outShape : numpy compatible shape tuple for input/output
#     """
#
#     info = {}
#
#     # I assume the model has only one input from the host (the first one). I
#     # suppose we could validate this somehow by parsing the graph but we're
#     # just gonna assume for now.
#     inNode = onnxModel.graph.input[0]
#     info['inName'] = inNode.name
#     info['inType'] = onnxTypes[inNode.type.tensor_type.elem_type]
#
#     info['inShape'] = []
#     for dim in inNode.type.tensor_type.shape.dim:
#         info['inShape'].append(dim.dim_value)
#
#     outs = []
#     for outNode in onnxModel.graph.output:
#         outInfo = {
#             'outName': outNode.name,
#             'outType': onnxTypes[outNode.type.tensor_type.elem_type]}
#
#         outInfo['outShape'] = []
#         for dim in outNode.type.tensor_type.shape.dim:
#             outInfo['outShape'].append(dim.dim_value)
#         outs.append(outInfo)
#
#     info['outputs'] = outs
#
#     return info


def getOnnx(onnxPath, outputDir):
    inputShapeMap = {"input_tensor:0": (1, 3, 224, 224)}
    modelName = 'resnet50'

    model = onnx.load(onnxPath)
    if inputShapeMap is not None:
        model = fixOnnxDim(model, inputShapeMap)

    mod, params = relay.frontend.from_onnx(model)
    with tvm.transform.PassContext(opt_level=3):
        module = relay.build(mod, tvm.target.cuda(), params=params)
    module.export_library(outputDir / "resnet50.so")

    # meta = infbench.model.getOnnxInfo(model)
    meta = getOnnxInfo(model)
    with open(outputDir / f"{modelName}.json", 'w') as f:
        json.dump(meta, f, indent=True)

    graphPath = outputDir / (modelName + "_graph.json")
    with open(graphPath, 'w') as f:
        f.write(module.get_graph_json())

    paramPath = outputDir / (modelName + "_params.pkl")
    with open(paramPath, 'wb') as f:
        pickle.dump([module.params['p' + str(i)].asnumpy() for i in range(len(module.params))], f)

    metaPath = outputDir / (modelName + "_meta.ptx")
    cudaLib = module.get_lib().imported_modules[0]
    cudaLib.save(str(metaPath))

    srcPath = outputDir / (modelName + "_source.cu")
    with open(srcPath, 'w') as f:
        f.write(cudaLib.get_source())


def loadGraph(outDir):
    graph = open(outDir / "resnet50_graph.json")
    return json.load(graph)


# This method is useful because the intermediate buffers in multi-kernel nodes
# aren't present in the graph, so this code is needed in 2 separate locations.
def getInfo(buf, graph):
    name = buf.name
    index = int(name)
    dtype = graph["attrs"]["dltype"][1][index]
    shape = graph["attrs"]["shape"][1][index]
    return dtype, shape


def loadParams(outDir):
    path = outDir / "resnet50_params.pkl"
    params = pickle.load(open(path, 'rb'))
    return {'p' + str(i): params[i] for i in range(len(params))}, params


def metaFromReq(req, graph):
    constants = []
    inputs = []
    outputs = []
    constMap = dict()
    inputMap = dict()

    for kern in req.kernels:
        for bufName, ioType in zip(kern.arguments, kern.ioTypes):
            buf = req.bufferMap[bufName]
            if not buf.ephemeral:
                dtype, shape = getInfo(buf, graph)
                if buf.const:
                    constMap[int(bufName)] = buf
                elif ioType == 'i':
                    inputMap[int(bufName)] = buf
            elif ioType == 'o':
                outputs.append({"name": buf.name, "type": dtype, "shape": shape})

    constant_list = list(constMap.keys())
    constant_list.sort()
    for idx, i in enumerate(constant_list):
        buf = constMap[i]
        dtype, shape = getInfo(buf, graph)
        constants.append({"name": buf.name, "type": dtype, "shape": shape, "dataIdx": idx})

    input_list = list(inputMap.keys())
    input_list.sort()
    for i in input_list:
        buf = inputMap[i]
        dtype, shape = getInfo(buf, graph)
        inputs.append({"name": buf.name, "type": dtype, "shape": shape})

    return {"constants": constants, "inputs": inputs, "outputs": outputs}


def getParams():
    params = loadParams()
    params_list = []
    for i in range(len(params.keys())):
        params_list.append(params["p" + str(i)])
    return params, params_list


def getSources(outDir):
    onnxPath = outDir / 'resnet50.onnx'
    if not onnxPath.exists():
        print("Downloading onnx")
        wget.download("https://zenodo.org/record/4735647/files/resnet50_v1.onnx", str(onnxPath))

    imagenetDir = outDir / 'fake_imagenet'
    if not imagenetDir.exists():
        print("Downloading input data")
        sp.run([modelDir / "make_fake_imagenet.sh", str(outDir)], check=True)


if __name__ == "__main__":
    if not outDir.exists():
        outDir.mkdir()

    getSources(outDir)

    getOnnx(outDir / 'resnet50.onnx', outDir)
    params_dict, params_list = loadParams(outDir)

    graph = loadGraph(outDir)

    sp.run(['make'], cwd=modelDir, check=True)

    req = createReq(params_dict, outDir / ("resnet50.cubin"))
    with open(outDir / "resnet50_model.pkl", 'wb') as f:
        pickle.dump(req, f)

    meta_data = metaFromReq(req, graph)
    with open(outDir / "resnet50_meta.yaml", 'w') as f:
        yaml.safe_dump(meta_data, f)

    with open(outDir / "resnet50_params.pkl", 'wb') as f:
        pickle.dump(params_list, f)
