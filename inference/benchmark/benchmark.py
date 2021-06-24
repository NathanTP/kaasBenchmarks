import pathlib
import filecmp

import infbench

import localBench
import rayBench

dataDir = (pathlib.Path(__file__).parent / ".." / "data").resolve()
modelDir = (pathlib.Path(__file__).parent / ".." / "models").resolve()


def sanityCheck():
    """Basic check to make sure nothing is obviously broken. This is meant to
    be manually fiddled with to spot check stuff. It will run the superres
    model and write the output to test.png, it should be the superres output (a
    small cat next to a big cat in a figure)."""
    bench = localBench
    # bench = rayBench

    bench.configure({"dataDir" : dataDir, "modelDir" : modelDir})
    res = bench.nShot("superRes", 1)

    with open("test.png", "wb") as f:
        f.write(res[0][0])

    print("Sanity check didn't crash!")
    print("Output available at ./test.png")


def nshot(modelName):
    # bench = localBench
    bench = rayBench
    bench.configure({"dataDir" : dataDir, "modelDir" : modelDir})
    res = bench.nShot(modelName, 16, inline=True)


def runMlperf(modelName):
    testing = False 
    inline = True 
    backend = "ray"

    if backend == 'ray':
        bench = rayBench
    elif backend == 'local':
        bench = localBench
    else:
        raise ArgumentError("Unrecognized backend: ", backend)

    bench.configure({"dataDir" : dataDir, "modelDir" : modelDir})

    print("Starting MLPerf Benchmark: ")
    print("\tModel: ", modelName)
    print("\tBackend: ", backend)
    print("\tTesting: ", testing)
    print("\tInline: ", inline)

    bench.mlperfBench(modelName, testing=testing, inline=inline)


def main():
    # sanityCheck()
    # nshot("resnet50")
    nshot("ssdMobilenet")
    # nshot("mobilenet-ssd")
    # runMlperf("superRes")
    # runMlperf("resnet50")
    # infbench.model.getOnnxInfo(modelDir / "resnet50.onnx")
    # infbench.model._loadOnnx(modelDir / "test.onnx", cache=False)

main()
