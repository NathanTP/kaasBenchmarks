import pathlib
import infbench.model
import subprocess as sp
import re
from pprint import pprint
import datetime
from kaas.pool import policies


clientUrl = "ipc://client.ipc"
barrierUrl = "ipc://barrier.ipc"

dataDir = (pathlib.Path(__file__).parent / ".." / "data").resolve()
modelDir = (pathlib.Path(__file__).parent / ".." / "models").resolve()


class ModelSpec():
    def __init__(self, name, loader, modelPath, modelClass, dataDir=dataDir,
                 modelType='native', impl='tvm', cacheInputs=False):
        self.name = name
        self.loader = loader
        self.dataDir = dataDir
        self.modelPath = modelPath
        self.modelClass = modelClass
        self.modelType = modelType
        self.impl = impl

        # This is a hack to deal with Ray's immutable object store. For models
        # with even modestly sized inputs, the benchmarks start crashing if you
        # generate a fresh input for every invocation. If cacheInputs is True,
        # we will re-use the same reference every time. This isn't ideal since
        # it will cut out a bunch of data layer contributions, but it will
        # affect both model types equally and it's the only way to avoid
        # crashes due to spilling to disk.
        self.cacheInputs = cacheInputs

    def getModelArg(self, constRefs=None, backend='ray'):
        if self.impl == 'tvm':
            return infbench.model.readModelBuf(self.modelPath)
        elif self.impl == 'kaas':
            # KaaS models live on the client so we only need one
            return self.modelClass(self.modelPath, constRefs, backend=backend)
        elif self.impl == "direct":
            return self.modelPath
        else:
            raise ValueError("Unrecognized model implementation method: ", self.impl)

    def getModelInstance(self, constRefs=None, backend='ray'):
        if self.impl == 'tvm':
            arg = infbench.model.readModelBuf(self.modelPath)
            return self.modelClass(arg)
        elif self.impl == 'kaas':
            return self.modelClass(self.modelPath, constRefs, backend=backend)
        elif self.impl == "direct":
            return self.modelClass(self.modelPath)
        else:
            raise ValueError("Unrecognized model implementation method: ", self.impl)


# This is implemented this way to ensure that models are only imported if
# necessary. Imports have a large impact on performance, and some models have a
# bigger impact than others.
def getModelSpec(modelName, modelType):
    # You must run tools/getModels.py first to get these .so's
    if modelName == "dummyModel" and modelType == "kaas":
        import infbench.dummyModel
        return ModelSpec(name="dummyModel",
                         loader=infbench.dummyModel.dummyLoader,
                         dataDir=modelDir / "dummy",
                         modelPath=modelDir / "dummy" / "dummy_model.yaml",
                         modelClass=infbench.dummyModel.dummyModelKaas,
                         modelType=modelType,
                         impl="kaas")

    elif modelName == "testModel" and modelType == "kaas":
        import infbench.testModel
        return ModelSpec(name="testModel",
                         loader=infbench.testModel.testLoader,
                         dataDir=modelDir / "sgemm",
                         modelPath=modelDir / "sgemm" / "sgemm_model.yaml",
                         modelClass=infbench.testModel.testModelKaas,
                         modelType=modelType,
                         impl="kaas")

    elif modelName == "testModel" and modelType == "native":
        import infbench.testModel
        return ModelSpec(name="testModel",
                         loader=infbench.testModel.testLoader,
                         dataDir=modelDir / "sgemm",
                         modelPath=modelDir / "sgemm" / "sgemm_meta.yaml",
                         modelClass=infbench.testModel.testModelNative,
                         modelType=modelType,
                         impl='direct')

    elif modelName == "jacobi" and modelType == "kaas":
        import infbench.jacobi
        return ModelSpec(name="jacobi",
                         loader=infbench.jacobi.jacobiLoader,
                         modelPath=modelDir / "jacobi" / "jacobi_model.yaml",
                         modelClass=infbench.jacobi.jacobiKaas,
                         modelType=modelType,
                         impl="kaas",
                         cacheInputs=True)

    elif modelName == "jacobi" and modelType == "native":
        import infbench.jacobi
        return ModelSpec(name="jacobi",
                         loader=infbench.jacobi.jacobiLoader,
                         modelPath=modelDir / "jacobi",
                         modelClass=infbench.jacobi.jacobi,
                         modelType=modelType,
                         impl="direct",
                         cacheInputs=True)

    elif modelName == "cGEMM" and modelType == "kaas":
        import infbench.cGEMM
        return ModelSpec(name="cGEMM",
                         loader=infbench.cGEMM.cutlassSgemmLoader,
                         dataDir=modelDir / "complexCutlassGemm",
                         modelPath=modelDir / "complexCutlassGemm" / "complexCutlassGemm_model.yaml",
                         modelClass=infbench.cGEMM.sgemmKaas,
                         modelType=modelType,
                         impl="kaas",
                         cacheInputs=True)

    elif modelName == "cGEMM" and modelType == "native":
        import infbench.cGEMM
        return ModelSpec(name="cGEMM",
                         loader=infbench.cGEMM.cutlassSgemmLoader,
                         dataDir=modelDir / "complexCutlassGemm",
                         modelPath=modelDir / "complexCutlassGemm" / "complexCutlassGemm_model.yaml",
                         modelClass=infbench.cGEMM.sgemm,
                         modelType=modelType,
                         impl="direct",
                         cacheInputs=True)

    elif modelName == "cutlassSgemm" and modelType == "kaas":
        import infbench.cutlassSgemm
        return ModelSpec(name="cutlassSgemm",
                         loader=infbench.cutlassSgemm.cutlassSgemmLoader,
                         modelPath=modelDir / "cutlassSgemm" / "cutlassSgemm_model.yaml",
                         modelClass=infbench.cutlassSgemm.sgemmKaas,
                         modelType=modelType,
                         impl="kaas")

    elif modelName == "cutlassSgemm" and modelType == "native":
        import infbench.cutlassSgemm
        return ModelSpec(name="cutlassSgemm",
                         loader=infbench.cutlassSgemm.cutlassSgemmLoader,
                         modelPath=modelDir / "cutlassSgemm" / "cutlassSgemm_model.yaml",
                         dataDir=modelDir / "cutlassSgemm",
                         modelClass=infbench.cutlassSgemm.sgemm,
                         modelType=modelType,
                         impl="direct")

    elif modelName == "superRes" and modelType == "kaas":
        import infbench.superres
        return ModelSpec(name="superRes",
                         loader=infbench.superres.superResLoader,
                         modelPath=modelDir / "superRes" / "superRes_model.yaml",
                         modelClass=infbench.superres.superResKaas,
                         modelType=modelType,
                         impl="kaas")

    elif modelName == "superRes" and modelType == "native":
        import infbench.superres
        return ModelSpec(name="superRes",
                         loader=infbench.superres.superResLoader,
                         modelPath=modelDir / "superRes" / "superres.so",
                         modelClass=infbench.superres.superResTvm,
                         modelType=modelType,
                         impl='tvm')

    elif modelName == "resnet50" and modelType == "kaas":
        import infbench.resnet50
        return ModelSpec(name="resnet50",
                         loader=infbench.resnet50.imageNetLoader,
                         modelPath=modelDir / "resnet50" / "resnet50_model.yaml",
                         modelClass=infbench.resnet50.resnet50Kaas,
                         modelType=modelType,
                         impl="kaas",
                         cacheInputs=True)

    elif modelName == "resnet50" and modelType == "native":
        import infbench.resnet50
        return ModelSpec(name="resnet50",
                         loader=infbench.resnet50.imageNetLoader,
                         modelPath=modelDir / "resnet50" / "resnet50.so",
                         modelClass=infbench.resnet50.resnet50,
                         modelType=modelType,
                         impl='tvm',
                         cacheInputs=True)

    elif modelName == "bert" and modelType == "kaas":
        import infbench.bert
        return ModelSpec(name="bert",
                         loader=infbench.bert.bertLoader,
                         modelClass=infbench.bert.bertModelKaas,
                         modelPath=modelDir / "bert" / "bert_model.yaml",
                         modelType=modelType,
                         impl="kaas")

    elif modelName == "bert" and modelType == "native":
        import infbench.bert
        return ModelSpec(name="bert",
                         loader=infbench.bert.bertLoader,
                         dataDir=dataDir,
                         modelPath=modelDir / 'bert' / "bert.so",
                         modelClass=infbench.bert.bertModel,
                         modelType=modelType,
                         impl='tvm')

    elif modelName == "ssdMobileNet" and modelType == "native":
        import infbench.ssdmobilenet
        return ModelSpec(name="ssdMobileNet",
                         loader=infbench.ssdmobilenet.cocoLoader,
                         modelPath=modelDir / "ssdMobilenet.so",
                         modelClass=infbench.ssdmobilenet.ssdMobilenet,
                         modelType=modelType,
                         impl='tvm'
                         )

    else:
        raise ValueError(f"Unrecognized model: {modelName} ({modelType})")


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


def getExpKey(benchConfig):
    """Return a unique identifier for the experiment configuration based on a
    benchconfig. Everything should use this when logging or analyzing
    experiments."""
    if benchConfig['modelType'] == 'kaas':
        return 'kaas'
    elif benchConfig['modelType'] == 'native':
        if benchConfig['policy'] == policies.EXCLUSIVE:
            return 'exclusive'
        elif benchConfig['policy'] == policies.STATIC and benchConfig['fractional'] is None:
            return 'static'
        elif benchConfig['policy'] == policies.STATIC and benchConfig['fractional'] is not None:
            return 'fractional'
        else:
            raise ValueError(f"Unrecognized or invalid configuration: {benchConfig['modelType']}-{benchConfig['policy']}")
    else:
        raise ValueError(f"Unrecognized or invalid configuration: {benchConfig['modelType']}-{benchConfig['policy']}")


def argsToConfig(args):
    """Generate a benchConfig from parseargs args. This relies on all
    benchmarks using the same names for common arguments."""
    if args.fractional is not None and args.policy != 'static':
        raise ValueError("'fractional' can only be used with the static policy")

    if args.policy == 'balance':
        args.policy = policies.BALANCE
    elif args.policy == 'exclusive':
        args.policy = policies.EXCLUSIVE
    elif args.policy == 'static':
        args.policy = policies.STATIC
    else:
        raise ValueError("Unsupported policy: ", args.policy)

    benchConfig = {
        "time": datetime.datetime.today().strftime("%d%m%y-%H%M%S"),
        "gitHash": currentGitHash()
    }

    benchConfig |= vars(args)
    benchConfig['expKey'] = getExpKey(benchConfig)

    return benchConfig
