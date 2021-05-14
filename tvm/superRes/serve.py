import numpy as np
import pathlib
import pickle
import tempfile
import os
import io

import onnx
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib.download import download_testdata
from tvm.contrib import graph_executor
from PIL import Image
import matplotlib.pyplot as plt


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
    libraryPath = pathlib.Path.cwd() / (onnxPath.stem + ".tvm.tar")

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


def getSuperRes():
    # Pre-Trained ONNX Model
    modelUrl = "".join(
        [
            "https://gist.github.com/zhreshold/",
            "bcda4716699ac97ea44f791c24310193/raw/",
            "93672b029103648953c4e5ad3ac3aadf346a4cdc/",
            "super_resolution_0.2.onnx",
        ]
    )
    modelPath = download_testdata(modelUrl, "super_resolution.onnx", module="onnx")
    return pathlib.Path(modelPath)


def imagePreprocess(buf):
    # mode and size were manually read from the png (used Image.open and then
    # inspected the img.mode and img.size attributes). We're gonna just go with
    # this for now.
    img = Image.frombytes("RGB", (256,256), buf)

    img = img.resize((224,224))
    img = img.convert("YCbCr")
    img_y, img_cb, img_cr = img.split()
    imgNp = (np.array(img_y)[np.newaxis, np.newaxis, :, :]).astype("float32")
    return (img, imgNp)


def imagePostProcess(imgPil, outNp):
    img_y, img_cb, img_cr = imgPil.split()
    out_y = Image.fromarray(outNp, mode="L")
    out_cb = img_cb.resize(out_y.size, Image.BICUBIC)
    out_cr = img_cr.resize(out_y.size, Image.BICUBIC)
    result = Image.merge("YCbCr", [out_y, out_cb, out_cr]).convert("RGB")
    canvas = np.full((672, 672 * 2, 3), 255)
    canvas[0:224, 0:224, :] = np.asarray(imgPil)
    canvas[:, 672:, :] = np.asarray(result)
    plt.imshow(canvas.astype(np.uint8))

    with io.BytesIO() as f:
        plt.savefig(f, format="png")
        pngBuf = f.getvalue()
    return pngBuf
    # plt.savefig("test.png", format="png")


def getImage():
    img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    img_path = download_testdata(img_url, "cat.png", module="data")
    img = Image.open(img_path)

    # We go through bytes to better align with faas requirements
    return img.tobytes()


def executeModel(ex, img):
    ex.set_input('1', tvm.nd.array(img))
    ex.run()
    tvm_output = ex.get_output(0, tvm.nd.empty((1, 1, 672, 672))).asnumpy()
    outNp = np.uint8((tvm_output[0, 0]).clip(0, 255))
    return outNp


def main():
    imgBuf = getImage()
    imgPil, imgNp = imagePreprocess(imgBuf)

    modelPath = getSuperRes()
    ex = importOnnx(modelPath, imgNp.shape)

    # with open("super_resolution.tvm.tar.so", 'rb') as f:
    #     modelBuf = f.read()
    # ex = importModelBuffer(modelBuf)

    # Execute
    print("Running model")
    out = executeModel(ex, imgNp)

    # Display Result
    print("Success, saving output to test.png")
    pngBuf = imagePostProcess(imgPil, out)

    with open("test.png", "wb") as f:
        f.write(pngBuf)

if __name__ == "__main__":
    main()
