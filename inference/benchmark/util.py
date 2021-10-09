import pathlib
import infbench.model
import subprocess as sp
import re
import os
from pprint import pprint


clientUrl = "ipc://client.ipc"
barrierUrl = "ipc://barrier.ipc"

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

    def getModelArg(self, constRefs = None):
        if self.modelType == 'tvm':
            return infbench.model.readModelBuf(self.modelPath)
        elif self.modelType == 'kaas':
            # KaaS models live on the client so we only need one
            return self.modelClass(self.modelPath, constRefs)
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

    elif modelName == "jacobi":
        import infbench.jacobi
        return ModelSpec(name="jacobi",
                         loader=infbench.jacobi.jacobiLoader,
                         modelPath = modelDir / "jacobi",
                         modelClass = infbench.jacobi.jacobi,
                         modelType="direct")


    elif modelName == "complexCutlassGemmKaas":
        import infbench.complexCutlassGemm
        return ModelSpec(name="complexCutlassGemm",
                         loader=infbench.complexCutlassGemm.cutlassSgemmLoader,
                         modelPath = modelDir / "complexCutlassGemm" / "complexCutlassGemm_model.yaml",
                         modelClass= infbench.complexCutlassGemm.sgemmKaas,
                        modelType="kaas")

    elif modelName == "complexCutlassGemm":
        import infbench.complexCutlassGemm
        return ModelSpec(name="complexCutlassGemm",
                         loader=infbench.complexCutlassGemm.cutlassSgemmLoader,
                         modelPath=modelDir / "complexCutlassGemm" / "complexCutlassGemm_model.yaml",
                         dataDir = modelDir / "complexCutlassGemm",
                         modelClass=infbench.complexCutlassGemm.sgemm,
                         modelType="direct")


    elif modelName == "cutlassSgemmKaas":
        import infbench.cutlassSgemm
        return ModelSpec(name="cutlassSgemmKaas",
                         loader=infbench.cutlassSgemm.cutlassSgemmLoader,
                         modelPath=modelDir / "cutlassSgemm" / "cutlassSgemm_model.yaml",
                         modelClass=infbench.cutlassSgemm.sgemmKaas,
                         modelType="kaas")
    elif modelName == "cutlassSgemm":
        import infbench.cutlassSgemm
        return ModelSpec(name="cutlassSgemm",
                         loader=infbench.cutlassSgemm.cutlassSgemmLoader,
                         modelPath=modelDir / "cutlassSgemm" / "cutlassSgemm_model.yaml",
                         dataDir = modelDir / "cutlassSgemm",
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
            if not isinstance(argMap, tuple):
                argMap = (argMap,)

            inputs.extend([data[i] for i in argMap])
    return inputs


def getGpuType():
    """Return a string describing the first available GPU"""
    proc = sp.run(['nvidia-smi', '-L'], text=True, stdout=sp.PIPE, check=True)
    match = re.search(r".*: (.*) \(UUID", proc.stdout)
    return match.group(1)


nGpu = None


def getNGpu():
    """Returns the number of available GPUs on this machine"""
    global nGpu
    if nGpu is None:
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            nGpu = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        else:
            proc = sp.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                          stdout=sp.PIPE, text=True, check=True)
            nGpu = proc.stdout.count('\n')

    return nGpu


def analyzeStats(stats):
    pat = re.compile("(.*\:)?t_.*")  # NOQA
    timeStats = {}
    otherStats = {}
    for m, v in stats.items():
        if pat.match(m):
            timeStats[m] = v['p50']
        else:
            otherStats[m] = v['p50']

    print("Time Stats:")
    pprint(timeStats)
    # kaasTimes = ['kaas:t_cudaMM', 'kaas:t_devDEvict', 'kaas:t_dtoh', 'kaas:t_hostDLoad', 'kaas:t_hostDWriteBack', 'kaas:t_hostMM', 'kaas:t_htod', 'kaas:t_invokeExternal', 'kaas:t_makeRoom', 'kaas:t_zero']
    # otherTimes = sum([timeStats[stat] for stat in kaasTimes])
    # print("Missing: ", timeStats['kaas:t_e2e'] - otherTimes)
    #
    # print("Other Stats:")
    # pprint(otherStats)


def mergePerClientStats(base, delta):
    for cID, deltaClient in delta.items():
        if cID in base:
            base[cID].merge(deltaClient)
        else:
            base[cID] = deltaClient


def currentGitHash():
    p = sp.run(['git', 'rev-parse', 'HEAD'], stdout=sp.PIPE, check=True, text=True, cwd=pathlib.Path(__file__).parent)
    return p.stdout.strip()
