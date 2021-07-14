#!/usr/bin/env python

import pathlib
import infbench.model

dataDir = (pathlib.Path(__file__).parent / ".." / "data").resolve()
modelDir = (pathlib.Path(__file__).parent / ".." / "models").resolve()


class ModelSpec():
    def __init__(self, name, loader, modelPath, modelClass, dataDir=dataDir, modelType='tvm'):
        self.name = name
        self.loader = loader
        self.dataDir = dataDir
        self.modelPath = modelPath
        self.modelClass = modelClass
        self.modelType = modelType

    def getModelArg(self):
        if self.modelType == 'tvm':
            return infbench.model.readModelBuf(self.modelPath)
        elif self.modelType == 'kaas':
            # KaaS models live on the client so we only need one
            return self.modelClass(self.modelPath)
        elif self.modelType == "direct":
            return None
        else:
            raise ValueError("Unrecognized model type: ", self.modelType)


# This is implemented this way to ensure that models are only imported if
# necessary. Imports have a large impact on performance, and some models have a
# bigger impact than others.
def getModelSpec(modelName):
    # You must run tools/getModels.py first to get these .so's
    if modelName == "testModelKaas":
        import infbench.testModel
        return ModelSpec(name="testModelKaas",
                         loader=infbench.testModel.testLoader,
                         modelPath=modelDir / "sgemm",
                         modelClass=infbench.testModel.testModelKaas,
                         modelType="kaas")

    elif modelName == "testModelNP":
        import infbench.testModel
        return ModelSpec(name="testModelNP",
                         loader=infbench.testModel.testLoader,
                         modelPath=modelDir,
                         modelClass=infbench.testModel.testModelNP,
                         modelType="direct")

    elif modelName == "superRes":
        import infbench.superres
        return ModelSpec(name="superRes",
                         loader=infbench.superres.superResLoader,
                         modelPath=modelDir / "superres.so",
                         modelClass=infbench.superres.superResTvm)

    elif modelName == "resnet50":
        import infbench.resnet50
        return ModelSpec(name="resnet50",
                         loader=infbench.resnet50.imageNetLoader,
                         modelPath=modelDir / "resnet50.so",
                         modelClass=infbench.resnet50.resnet50)

    elif modelName == "ssdMobileNet":
        import infbench.ssdmobilenet
        return ModelSpec(name="ssdMobileNet",
                         loader=infbench.ssdmobilenet.cocoLoader,
                         modelPath=modelDir / "ssdMobilenet.so",
                         modelClass=infbench.ssdmobilenet.ssdMobilenet)

    elif modelName == "bert":
        import infbench.bert
        return ModelSpec(name="bert",
                         loader=infbench.bert.bertLoader,
                         dataDir=dataDir,
                         modelPath=modelDir / 'bert' / "bert.so",
                         modelClass=infbench.bert.bertModel)

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
    inline = False

    print("Starting MLPerf Benchmark: ")
    print("\tModel: ", modelSpec.name)
    print("\tBackend: ", backend.__name__)
    print("\tTesting: ", testing)
    print("\tInline: ", inline)

    backend.mlperfBench(modelSpec, testing=testing, inline=inline)


def main():
    spec = getModelSpec("testModelKaas")

    # import localBench
    # backend = localBench

    import rayBench
    backend = rayBench

    # sanityCheck()
    nshot(spec, 1, backend)
    # runMlperf(spec, backend)


main()
