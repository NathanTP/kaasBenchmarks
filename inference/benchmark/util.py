import pathlib
import infbench.model


clientUrl = "ipc://benchmark_client.ipc"

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
            return self.modelPath
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
                         modelPath=modelDir / "sgemm" / "sgemm_model.yaml",
                         modelClass=infbench.testModel.testModelKaas,
                         modelType="kaas")

    elif modelName == "superResKaas":
        import infbench.superres
        return ModelSpec(name="superResKaas",
                         loader=infbench.superres.superResLoader,
                         modelPath=modelDir / "superRes" / "superRes_model.yaml",
                         modelClass=infbench.superres.superResKaas,
                         modelType="kaas")

    elif modelName == "resnet50Kaas":
        import infbench.resnet50
        return ModelSpec(name="resnet50Kaas",
                         loader=infbench.resnet50.imageNetLoader,
                         modelPath=modelDir / "resnet50" / "resnet50_model.yaml",
                         modelClass=infbench.resnet50.resnet50Kaas,
                         modelType="kaas")

    elif modelName == "bertKaas":
        import infbench.bert
        return ModelSpec(name="bertKaas",
                         loader=infbench.bert.bertLoader,
                         modelClass=infbench.bert.bertModelKaas,
                         modelPath=modelDir / "bert" / "bert_model.yaml",
                         modelType="kaas")

    elif modelName == "testModelNP":
        import infbench.testModel
        return ModelSpec(name="testModelNP",
                         loader=infbench.testModel.testLoader,
                         modelPath=modelDir,  # testModelNP is completely self-contained, modelDir is unused
                         modelClass=infbench.testModel.testModelNP,
                         modelType="direct")

    elif modelName == "testModelNative":
        import infbench.testModel
        return ModelSpec(name="testModelNative",
                         loader=infbench.testModel.testLoader,
                         modelPath=modelDir / "sgemm" / "sgemm_meta.yaml",
                         modelClass=infbench.testModel.testModelNative,
                         modelType="direct")

    elif modelName == "superRes":
        import infbench.superres
        return ModelSpec(name="superRes",
                         loader=infbench.superres.superResLoader,
                         modelPath=modelDir / "superRes" / "superres.so",
                         modelClass=infbench.superres.superResTvm)

    elif modelName == "resnet50":
        import infbench.resnet50
        return ModelSpec(name="resnet50",
                         loader=infbench.resnet50.imageNetLoader,
                         modelPath=modelDir / "resnet50" / "resnet50.so",
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
