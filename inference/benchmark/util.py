import pathlib
import infbench.model
import subprocess as sp
import re
from pprint import pprint


clientUrl = "ipc://client.ipc"
barrierUrl = "ipc://barrier.ipc"

dataDir = (pathlib.Path(__file__).parent / ".." / "data").resolve()
modelDir = (pathlib.Path(__file__).parent / ".." / "models").resolve()


class ModelSpec():
    def __init__(self, name, loader, modelPath, modelClass, dataDir=dataDir, modelType='tvm', cacheInputs=False):
        self.name = name
        self.loader = loader
        self.dataDir = dataDir
        self.modelPath = modelPath
        self.modelClass = modelClass
        self.modelType = modelType

        # This is a hack to deal with Ray's immutable object store. For models
        # with even modestly sized inputs, the benchmarks start crashing if you
        # generate a fresh input for every invocation. If cacheInputs is True,
        # we will re-use the same reference every time. This isn't ideal since
        # it will cut out a bunch of data layer contributions, but it will
        # affect both model types equally and it's the only way to avoid
        # crashes due to spilling to disk.
        self.cacheInputs = cacheInputs

    def getModelArg(self, constRefs=None, backend='ray'):
        if self.modelType == 'tvm':
            return infbench.model.readModelBuf(self.modelPath)
        elif self.modelType == 'kaas':
            # KaaS models live on the client so we only need one
            return self.modelClass(self.modelPath, constRefs, backend=backend)
        elif self.modelType == "direct":
            return self.modelPath
        else:
            raise ValueError("Unrecognized model type: ", self.modelType)

    def getModelInstance(self, constRefs=None, backend='ray'):
        if self.modelType == 'tvm':
            arg = infbench.model.readModelBuf(self.modelPath)
            return self.modelClass(arg)
        elif self.modelType == 'kaas':
            return self.modelClass(self.modelPath, constRefs, backend=backend)
        elif self.modelType == "direct":
            return self.modelClass(self.modelPath)
        else:
            raise ValueError("Unrecognized model type: ", self.modelType)


# This is implemented this way to ensure that models are only imported if
# necessary. Imports have a large impact on performance, and some models have a
# bigger impact than others.
def getModelSpec(modelName):
    # You must run tools/getModels.py first to get these .so's
    if modelName == "dummyModelKaas":
        import infbench.dummyModel
        return ModelSpec(name="dummyModelKaas",
                         loader=infbench.dummyModel.dummyLoader,
                         dataDir=modelDir / "dummy",
                         modelPath=modelDir / "dummy" / "dummy_model.yaml",
                         modelClass=infbench.dummyModel.dummyModelKaas,
                         modelType="kaas")

    elif modelName == "testModelKaas":
        import infbench.testModel
        return ModelSpec(name="testModelKaas",
                         loader=infbench.testModel.testLoader,
                         dataDir=modelDir / "sgemm",
                         modelPath=modelDir / "sgemm" / "sgemm_model.yaml",
                         modelClass=infbench.testModel.testModelKaas,
                         modelType="kaas")

    elif modelName == "jacobiTvm":
        import infbench.jacobi
        return ModelSpec(name="jacobi",
                         loader=infbench.jacobi.jacobiLoader,
                         modelPath=modelDir / "jacobi",
                         modelClass=infbench.jacobi.jacobi,
                         modelType="direct")

    elif modelName == "jacobiKaas":
        import infbench.jacobi
        return ModelSpec(name="jacobiKaas",
                         loader=infbench.jacobi.jacobiLoader,
                         modelPath=modelDir / "jacobi" / "jacobi_model.yaml",
                         modelClass=infbench.jacobi.jacobiKaas,
                         modelType="kaas")

    elif modelName == "cGEMMKaas":
        import infbench.cGEMM
        return ModelSpec(name="cGEMM",
                         loader=infbench.cGEMM.cutlassSgemmLoader,
                         dataDir=modelDir / "complexCutlassGemm",
                         modelPath=modelDir / "complexCutlassGemm" / "complexCutlassGemm_model.yaml",
                         modelClass=infbench.cGEMM.sgemmKaas,
                         modelType="kaas",
                         cacheInputs=True)

    elif modelName == "cGEMMTvm":
        import infbench.cGEMM
        return ModelSpec(name="cGEMM",
                         loader=infbench.cGEMM.cutlassSgemmLoader,
                         dataDir=modelDir / "complexCutlassGemm",
                         modelPath=modelDir / "complexCutlassGemm" / "complexCutlassGemm_model.yaml",
                         modelClass=infbench.cGEMM.sgemm,
                         modelType="direct",
                         cacheInputs=True)

    elif modelName == "cutlassSgemmKaas":
        import infbench.cutlassSgemm
        return ModelSpec(name="cutlassSgemmKaas",
                         loader=infbench.cutlassSgemm.cutlassSgemmLoader,
                         modelPath=modelDir / "cutlassSgemm" / "cutlassSgemm_model.yaml",
                         modelClass=infbench.cutlassSgemm.sgemmKaas,
                         modelType="kaas")

    elif modelName == "cutlassSgemmTvm":
        import infbench.cutlassSgemm
        return ModelSpec(name="cutlassSgemm",
                         loader=infbench.cutlassSgemm.cutlassSgemmLoader,
                         modelPath=modelDir / "cutlassSgemm" / "cutlassSgemm_model.yaml",
                         dataDir=modelDir / "cutlassSgemm",
                         modelClass=infbench.cutlassSgemm.sgemm,
                         modelType="direct")

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

    elif modelName == "testModelTvm":
        import infbench.testModel
        return ModelSpec(name="testModelNative",
                         loader=infbench.testModel.testLoader,
                         dataDir=modelDir / "sgemm",
                         modelPath=modelDir / "sgemm" / "sgemm_meta.yaml",
                         modelClass=infbench.testModel.testModelNative,
                         modelType="direct")

    elif modelName == "superResTvm":
        import infbench.superres
        return ModelSpec(name="superRes",
                         loader=infbench.superres.superResLoader,
                         modelPath=modelDir / "superRes" / "superres.so",
                         modelClass=infbench.superres.superResTvm)

    elif modelName == "resnet50Tvm":
        import infbench.resnet50
        return ModelSpec(name="resnet50",
                         loader=infbench.resnet50.imageNetLoader,
                         modelPath=modelDir / "resnet50" / "resnet50.so",
                         modelClass=infbench.resnet50.resnet50)

    elif modelName == "ssdMobileNetTvm":
        import infbench.ssdmobilenet
        return ModelSpec(name="ssdMobileNet",
                         loader=infbench.ssdmobilenet.cocoLoader,
                         modelPath=modelDir / "ssdMobilenet.so",
                         modelClass=infbench.ssdmobilenet.ssdMobilenet)

    elif modelName == "bertTvm":
        import infbench.bert
        return ModelSpec(name="bert",
                         loader=infbench.bert.bertLoader,
                         dataDir=dataDir,
                         modelPath=modelDir / 'bert' / "bert.so",
                         modelClass=infbench.bert.bertModel)

    else:
        raise ValueError("Unrecognized model: ", modelName)


def packInputs(maps, const=None, inp=None, pre=None, run=None):
    inputs = []
    for (argMap, data) in zip(maps, [const, inp, pre, run]):
        if argMap is not None:
            if data is None:
                continue

            if isinstance(argMap, int):
                argMap = [argMap]
            inputs.extend([data[i] for i in argMap])
    return inputs


def analyzeStats(stats):
    pat = re.compile("(.*\:)?t_.*")  # NOQA
    timeStats = {}
    otherStats = {}
    for m, v in stats.items():
        if pat.match(m):
            timeStats[m] = v['mean']
        else:
            otherStats[m] = v['mean']

    print("Time Stats:")
    pprint(timeStats)
    # kaasTimes = ['kaas:t_cudaMM', 'kaas:t_devDEvict', 'kaas:t_dtoh', 'kaas:t_hostDLoad', 'kaas:t_hostDWriteBack', 'kaas:t_hostMM', 'kaas:t_htod', 'kaas:t_invokeExternal', 'kaas:t_makeRoom', 'kaas:t_zero']
    # otherTimes = sum([timeStats[stat] for stat in kaasTimes])
    # print("Missing: ", timeStats['kaas:t_e2e'] - otherTimes)
    #
    # print("Other Stats:")
    # pprint(otherStats)


def currentGitHash():
    p = sp.run(['git', 'rev-parse', 'HEAD'], stdout=sp.PIPE, check=True, text=True, cwd=pathlib.Path(__file__).parent)
    return p.stdout.strip()
