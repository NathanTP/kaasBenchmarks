import infbench
import pathlib

import localBench
import rayBench

dataDir = (pathlib.Path(__file__).parent / ".." / "data").resolve()
modelDir = (pathlib.Path(__file__).parent / ".." / "models").resolve()

def getSuperres():
    loader = infbench.dataset.superResLoader(dataDir / "superres" / "cat.png")
    dataProc = infbench.dataset.superResProcessor()
    model = infbench.model.superRes(modelDir / "super_resolution.onnx")

    return (loader, dataProc, model)


def sanityCheck():
    """Basic check to make sure nothing is obviously broken. This is meant to
    be manually fiddled with to spot check stuff. It will run the superres
    model and write the output to test.png, it should be the superres output (a
    small cat next to a big cat in a figure)."""
    # localBench.configure({"dataDir" : dataDir, "modelDir" : modelDir})
    # res = localBench.oneShot("superRes")

    rayBench.configure({"dataDir" : dataDir, "modelDir" : modelDir})
    res = rayBench.oneShot("superRes")

    with open("test.png", "wb") as f:
        f.write(res)


def main():
    sanityCheck()

main()
