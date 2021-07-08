import pathlib

dataDir = (pathlib.Path(__file__).parent / ".." / "data").resolve()
modelDir = (pathlib.Path(__file__).parent / ".." / "models").resolve()


# This is implemented this way to ensure that models are only imported if
# necessary. Imports have a large impact on performance, and some models have a
# bigger impact than others.
def getModelSpec(modelName):
    # You must run tools/getModels.py first to get these .so's
    if modelName == "testModel":
        import infbench.testModel
        return {
            "name": "testModel",
            "loader": infbench.testModel.testLoader,
            "dataDir": dataDir,
            "modelPath": modelDir,
            "modelClass": infbench.testModel.testModel
        }
    elif modelName == "superRes":
        import infbench.superres
        return {
            "name": "superRes",
            "loader": infbench.superres.superResLoader,
            "dataDir": dataDir,
            "modelPath": modelDir / "superres.so",
            "modelClass": infbench.superres.superRes
        }
    elif modelName == "resnet50":
        import infbench.resnet50
        return {
            "name": "resnet50",
            "loader": infbench.resnet50.imageNetLoader,
            "dataDir": dataDir,
            "modelPath": modelDir / "resnet50.so",
            "modelClass": infbench.resnet50.resnet50
        }
    elif modelName == "ssdMobileNet":
        import infbench.ssdmobilenet
        return {
            "name": "ssdMobileNet",
            "loader": infbench.ssdmobilenet.cocoLoader,
            "dataDir": dataDir,
            "modelPath": modelDir / "ssdMobilenet.so",
            "modelClass": infbench.ssdmobilenet.ssdMobilenet
        }
    elif modelName == "bert":
        import infbench.bert
        return {
            "name": "bert",
            "loader": infbench.bert.bertLoader,
            "dataDir": dataDir,
            "modelPath": modelDir / 'bert' / "bert.so",
            "modelClass": infbench.bert.bertModel
        }
    else:
        raise ValueError("Unrecognized model: ", modelName)


def sanityCheck(backend):
    """Basic check to make sure nothing is obviously broken. This is meant to
    be manually fiddled with to spot check stuff. It will run the superres
    model and write the output to test.png, it should be the superres output (a
    small cat next to a big cat in a figure)."""
    spec = getModelSpec("superRes")
    res = backend.nShot(spec, 1)

    with open("test.png", "wb") as f:
        f.write(res[0][0])

    print("Sanity check didn't crash!")
    print("Output available at ./test.png")


def nshot(modelSpec, n, backend):
    backend.nShot(modelSpec, n, inline=False)


def runMlperf(modelSpec, backend):
    testing = True
    inline = True

    print("Starting MLPerf Benchmark: ")
    print("\tModel: ", modelSpec['name'])
    print("\tBackend: ", backend.__name__)
    print("\tTesting: ", testing)
    print("\tInline: ", inline)

    backend.mlperfBench(modelSpec, testing=testing, inline=inline)


def main():
    spec = getModelSpec("testModel")

    import localBench
    backend = localBench

    # import rayBench
    # backend = rayBench

    # sanityCheck()
    nshot(spec, 1, backend)
    # runMlperf(spec, backend)


main()
