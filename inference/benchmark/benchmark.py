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
    # bench = localBench
    bench = rayBench

    bench.configure({"dataDir" : dataDir, "modelDir" : modelDir})
    # res = bench.oneShot("superRes", inline=False)
    res = bench.oneShot("superRes")

    with open("test.png", "wb") as f:
        f.write(res)

    print("Sanity check didn't crash!")
    if filecmp.cmp("test.png", dataDir / "superRes" / "catSupered.png"):
        print("Result looks reasonable")
    else:
        print("Result doesn't look right. Check test.png.")


def nshot(modelName):
    bench = localBench
    bench.configure({"dataDir" : dataDir, "modelDir" : modelDir})
    res = bench.nShot(modelName, 8)


def runMlperf(modelName):
    bench = rayBench
    # bench = localBench
    bench.configure({"dataDir" : dataDir, "modelDir" : modelDir})
    bench.mlperfBench(modelName, testing=False)


def main():
    # sanityCheck()
    nshot("resnet50")
    # runMlperf()
    # infbench.model.getOnnxInfo(modelDir / "resnet50.onnx")

main()
