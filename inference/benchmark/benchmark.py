import infbench
import pathlib

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
    res = bench.oneShot("superRes")

    # rayBench.configure({"dataDir" : dataDir, "modelDir" : modelDir})
    # res = rayBench.oneShot("superRes")

    with open("test.png", "wb") as f:
        f.write(res)



def nshot():
    bench = localBench
    bench.configure({"dataDir" : dataDir, "modelDir" : modelDir})
    res = bench.nShot("superRes", 16)


def runMlperf():
    localBench.configure({"dataDir" : dataDir, "modelDir" : modelDir})
    localBench.mlperfBench("superRes")


def main():
    print("Starting Benchmark:")
    # sanityCheck()
    # nshot()
    runMlperf()

main()
