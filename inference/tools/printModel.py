import infbench
import sys
import pathlib
import argparse
from pprint import pprint
import onnx

parser = argparse.ArgumentParser()
parser.add_argument("model", help="Path to model")
parser.add_argument("-d", "--details", metavar="DESTINATION", help="Store detailed information in the directory DESTINATION")
args = parser.parse_args()

modelPath = pathlib.Path(args.model)

onnxModel = onnx.load(modelPath)
onnx.checker.check_model(onnxModel)
# print(onnxModel)

meta = infbench.model.getOnnxInfo(onnxModel)
print("Model Metadata:")
pprint(meta)


# model, meta = infbench.model._loadOnnx(modelPath, cache=False)
# print("Model loaded successfully:")
# pprint(meta)

# if args.details is not None:
#     outDir = pathlib.Path(args.details).resolve()
#     print("Dumping detailed output to ", outDir) 
#     dumpModel(model, outDir)
