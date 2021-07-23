#!/usr/bin/env python
import util

def sanityCheck(backend):
    """Basic check to make sure nothing is obviously broken. This is meant to
    be manually fiddled with to spot check stuff. It will run the superres
    model and write the output to test.png, it should be the superres output (a
    small cat next to a big cat in a figure)."""
    spec = util.getModelSpec("superRes")
    res = backend.nShot(spec, 1)

    with open("test.png", "wb") as f:
        f.write(res[0][0])

    print("Sanity check didn't crash!")
    print("Output available at ./test.png")


def nshot(modelSpec, n, backend):
    backend.nShot(modelSpec, n, inline=False, useActors=True)


def runMlperf(modelSpec, backend):
    testing = False
    inline = False

    print("Starting MLPerf Benchmark: ")
    print("\tModel: ", modelSpec.name)
    print("\tBackend: ", backend.__name__)
    print("\tTesting: ", testing)
    print("\tInline: ", inline)

    backend.mlperfBench(modelSpec, testing=testing, inline=inline, useActors=False)


def main():
    spec = util.getModelSpec("resnet50")

    # import localBench
    # backend = localBench

    import rayBench
    backend = rayBench
    # rayBench.serveRequests(useActors=True)

    # sanityCheck()
    nshot(spec, 16, backend)
    # runMlperf(spec, backend)


main()
