import pathlib
import infbench.model
import subprocess as sp
import re
import collections.abc
import contextlib
import time
import json
import numpy as np
import os


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

    elif modelName == "cutlassSgemmKaas":
        import infbench.cutlass
        return ModelSpec(name="cutlassSgemmKaas",
                         loader=infbench.cutlass.cutlassSgemmLoader,
                         modelPath=modelDir / "cutlassSgemm" / "cutlassSgemm_model.yaml",
                         modelClass=infbench.cutlass.sgemmKaas,
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
            assert data is not None
            inputs.extend([data[i] for i in argMap])
    return inputs


def getGpuType():
    """Return a string describing the first available GPU"""
    proc = sp.run(['nvidia-smi', '-L'], text=True, stdout=sp.PIPE, check=True)
    match = re.search(r".*: (.*) \(UUID", proc.stdout)
    return match.group(1)


class prof():
    def __init__(self, fromDict=None, detail=True):
        """A profiler object for a metric or event type. The counter can be
        updated multiple times per event, while calling increment() moves on to
        a new event. If detail==true, all events are logged allowing more
        complex statistics. This may affect performance if there are many
        events."""
        self.detail = detail
        if fromDict is not None:
            if self.detail:
                self.events = fromDict['events']
            self.total = fromDict['total']
            self.nevent = fromDict['nevent']
        else:
            self.total = 0.0
            self.nevent = 0

            if self.detail:
                self.currentEvent = 0.0
                self.events = []

    def update(self, n):
        """Update increases the value of this entry for the current event."""
        self.total += n
        if self.detail:
            self.currentEvent += n

    def increment(self, n=0):
        """Finalize the current event (increment the event counter). If n is
        provided, the current event will be updated by n before finalizing."""
        self.update(n)
        self.nevent += 1
        if self.detail:
            self.events.append(self.currentEvent)
            self.currentEvent = 0.0

    def report(self):
        """Report the average value per event"""
        rep = {}
        rep['total'] = self.total
        rep['mean'] = self.total / self.nevent
        if self.detail:
            events = np.array(self.events)
            rep['min'] = events.min()
            rep['max'] = events.max()
            rep['p50'] = np.quantile(events, 0.50)
            rep['p90'] = np.quantile(events, 0.90)
            rep['p99'] = np.quantile(events, 0.99)
            rep['events'] = self.events

        return rep


class profCollection(collections.abc.MutableMapping):
    """This is basically a dictionary and can be used anywhere a dictionary of
    profs was previously used. It has a few nice additional features though. In
    particular, it will generate an empty prof whenever a non-existant key is
    accessed."""

    def __init__(self, detail=True):
        # a map of modules included in these stats. Each module is a
        # profCollection. Submodules can nest indefinitely.
        self.detail = detail

        self.mods = {}

        self.profs = dict()

    def __getitem__(self, key):
        if key not in self.profs:
            self.profs[key] = prof(detail=self.detail)
        return self.profs[key]

    def __setitem__(self, key, value):
        self.profs[key] = value

    def __delitem__(self, key):
        del self.profs[key]

    def __iter__(self):
        return iter(self.profs)

    def __len__(self):
        return len(self.profs)

    def __str__(self):
        return json.dumps(self.report(), indent=4)

    def mod(self, name):
        if name not in self.mods:
            self.mods[name] = profCollection(detail=self.detail)

        return self.mods[name]

    def merge(self, new, prefix=''):
        # Start by merging the direct stats
        for k, v in new.items():
            newKey = prefix+k
            if newKey in self.profs:
                self.profs[newKey].increment(v.total)
            else:
                self.profs[newKey] = v

        # Now recursively handle modules
        for name, mod in new.mods.items():
            # Merging into an empty profCollection causes a deep copy
            if name not in self.mods:
                self.mods[name] = profCollection(detail=self.detail)
            self.mods[name].merge(mod)

    def report(self):
        flattened = {name: v.report() for name, v in self.profs.items()}

        for name, mod in self.mods.items():
            flattened = {**flattened, **{name+":"+itemName: v for itemName, v in mod.report().items()}}

        return flattened

    def reset(self):
        """Clears all existing metrics. Any instantiated modules will continue
        to exist, but will be empty (it is safe to keep references to modules
        after reset()).
        """
        self.profs = {}
        for mod in self.mods.values():
            mod.reset()


@contextlib.contextmanager
def timer(name, timers, final=True):
    if timers is None:
        yield
    else:
        start = time.time()
        try:
            yield
        finally:
            if final:
                timers[name].increment((time.time()) - start)
            else:
                timers[name].update((time.time()) - start)


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
