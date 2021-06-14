import pathlib
import wget
import subprocess as sp

modelDir = pathlib.Path("./models").resolve()

if not modelDir.exists():
    modelDir.mkdir(mode=0o700)

# These sources were taken from:
# https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection
# superres is from a tvm tutorial
models = {
        "resnet50" : "https://zenodo.org/record/4735647/files/resnet50_v1.onnx",
        "ssd-mobilenet" : "https://zenodo.org/record/4735652/files/ssd_mobilenet_v1_coco_2018_01_28.onnx",
        "ssd-resnet34" : "https://zenodo.org/record/4735664/files/ssd_resnet34_mAP_20.2.onnx",
        "superres" : "https://gist.github.com/zhreshold/bcda4716699ac97ea44f791c24310193/raw/93672b029103648953c4e5ad3ac3aadf346a4cdc/super_resolution_0.2.onnx"
}

for name,url in models.items():
    modelPath = modelDir / (name+".onnx")
    if not modelPath.exists():
        print("Downloading ", modelPath)
        wget.download(url, str(modelPath))


print("Fixing resnet50 onnx model:")
# resnet50 has some dynamic input shapes, this messes up TVM since it can't
# infer them for the static graph. The solution is to use this onnxsim
# (onnx-simplifier) to manually specify the shape and then simplify the onnx to
# make it all static. I believe that the leading 1 in the shape is the batch size.
sp.run(["python", '-m', 'onnxsim', '--input-shape=1,3,224,224', 'resnet50.onnx', 'resnet50.onnx'], cwd=modelDir)
