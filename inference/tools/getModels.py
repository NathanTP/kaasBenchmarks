import pathlib
import wget
import json
import onnx
import tvm
import tvm.relay as relay

# Gluoncv throws out some stupid warning about having both mxnet and torch,
# have to go through this nonsense to suppress it.
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import gluoncv.model_zoo

import infbench

modelDir = pathlib.Path(__file__).parent.resolve().parent / "models"


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


# This function was written with a bit of trial and error. the onnxModel
# returned by onnx.load is a protobuf structure, you can print it and inspect
# different parts of it.
def getOnnxInfo(onnxModel):
    """Get metadata from an onnx object. Metadata returned includes at least:
        - inName/outName   : Name of the input/output node
        - inType/outType   : numpy type of the input/output
        - inShape/outShape : numpy compatible shape tuple for input/output
    """

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

    outs = []
    for outNode in onnxModel.graph.output:
        outInfo = {
            'outName': outNode.name,
            'outType': onnxTypes[outNode.type.tensor_type.elem_type]}

        outInfo['outShape'] = []
        for dim in outNode.type.tensor_type.shape.dim:
            outInfo['outShape'].append(dim.dim_value)
        outs.append(outInfo)

    info['outputs'] = outs

    return info


def getOnnx(inputPath, outputDir, inputShapeMap=None):
    model = onnx.load(inputPath)
    if inputShapeMap is not None:
        model = fixOnnxDim(model, inputShapeMap)

    mod, params = relay.frontend.from_onnx(model)
    with tvm.transform.PassContext(opt_level=3):
        module = relay.build(mod, tvm.target.cuda(), params=params)
    module.export_library((outputDir / inputPath.name).with_suffix(".so"))

    meta = infbench.model.getOnnxInfo(model)
    with open((outputDir / inputPath.name).with_suffix(".json"), 'w') as f:
        json.dump(meta, f, indent=True)


def getResnet50():
    modelPath = modelDir / 'resnet50.onnx'
    if not modelPath.exists():
        wget.download("https://zenodo.org/record/4735647/files/resnet50_v1.onnx", str(modelPath))
    getOnnx(modelPath, modelDir, inputShapeMap={"input_tensor:0": (1, 3, 224, 224)})


def getSuperRes():
    modelPath = modelDir / 'superres.onnx'
    if not modelPath.exists():
        wget.download("https://gist.github.com/zhreshold/bcda4716699ac97ea44f791c24310193/raw/93672b029103648953c4e5ad3ac3aadf346a4cdc/super_resolution_0.2.onnx", str(modelPath))
    getOnnx(modelPath, modelDir)


def getBert():
    bertDir = modelDir / 'bert'
    modelPath = bertDir / 'bert.onnx'
    vocabPath = bertDir / 'vocab.txt'

    if not bertDir.exists():
        bertDir.mkdir()

    if not modelPath.exists():
        print("Downloading BERT model")
        wget.download("https://zenodo.org/record/3733910/files/model.onnx", str(modelPath))
    if not vocabPath.exists():
        print("Downloading BERT vocab")
        wget.download("https://zenodo.org/record/3733910/files/vocab.txt", str(vocabPath))

    print("Converting BERT to .so")
    getOnnx(modelPath, bertDir,
            inputShapeMap={
                'input_ids': (1, 384),
                'input_mask': (1, 384),
                'segment_ids': (1, 384)})


def getSsdMobilenet():
    block = gluoncv.model_zoo.get_model("ssd_512_mobilenet1.0_coco", pretrained=True)
    mod, params = relay.frontend.from_mxnet(block, {"data": (1, 3, 512, 512)})
    with tvm.transform.PassContext(opt_level=3):
        module = relay.build(mod, tvm.target.cuda(), params=params)
    module.export_library(modelDir / 'ssdMobilenet.so')

    # I'm sure there's a principled way to do this from mxnet models, but whatever
    meta = {
        "inputs": [
            {
                "name": "data",
                "type": 'float32',
                "shape": (1, 3, 512, 512)
            }
        ],
        "outputs": [
            {
                "outName": "classIDs",
                "outType": "float32",
                "outShape": (1, 100, 1),
            },
            {
                "outname": "scores",
                "outtype": "float32",
                "outshape": (1, 100, 1),
            },
            {
                "outname": "bboxes",
                "outtype": "float32",
                "outshape": (1, 100, 4),
            }
        ]
    }
    with open(modelDir / 'ssdMobilenet.json', 'w') as f:
        json.dump(meta, f)


def main():
    if not modelDir.exists():
        modelDir.mkdir(mode=0o700)

    print("Getting BERT")
    getBert()

    print("\nGetting Resnet")
    getResnet50()

    print("\nGetting SuperRes")
    getSuperRes()

    print("\nGetting SSD-Mobilenet")
    getSsdMobilenet()


main()
