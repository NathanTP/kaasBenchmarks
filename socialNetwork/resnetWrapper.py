import infbench
from infbench import resnet50
import kaas
from kaas import profiling

import pathlib
import ray
import sys


modelDir = pathlib.Path("resnetModel/resnet50")


@kaas.pool.remote_with_confirmation()
def _pre(args):
    preProcessed = resnet50.resnet50Kaas.pre(args)
    return preProcessed[0]


@ray.remote(num_gpus=1)
class TvmWorker(kaas.pool.PoolWorker):
    """A PoolWorker for running resnet50 TVM model requests."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.modelCache = {}
        self.model = resnet50.resnet50(modelDir / 'resnet50.so')

    def run(self, inputRefs, clientID=None):
        profs = self.getProfs()

        with profiling.timer('t_loadInput', profs):
            inputs = ray.get(inputRefs)

        with profiling.timer('t_model_run', profs):
            result = self.model.run(list(inputs), stats=profs)
        assert isinstance(result, tuple)

        with profiling.timer('t_writeOutput', profs):
            resRefs = [ray.put(res) for res in result]

        return tuple(resRefs)

    def shutdown(self):
        for modelInfo in self.modelCache.values():
            modelInfo[0].shutdown()


class ResnetHandle():
    def __init__(self, modelDir: pathlib.Path, mode: str):
        """Create a handle for interacting with resnet.
        Args:
            modelDir: Location of model files
            mode: Either 'kaas' or 'tvm'
        """
        self.mode = mode

        self.loader = resnet50.imageNetLoader(modelDir)
        self.loader.preLoad(range(self.loader.ndata))

        constants = resnet50.resnet50Kaas.getConstants(modelDir)
        self.constRefs = [ray.put(const) for const in constants]

        if self.mode == 'kaas':
            self.model = resnet50.resnet50Kaas(modelDir / 'resnet50_model.yaml',
                                               self.constRefs, backend='ray')
        elif self.mode == 'tvm':
            self.model = resnet50.resnet50(modelDir / 'resnet50.so')
        else:
            raise ValueError("Unrecognized mode: ", self.mode)

    def getInput(self, idx):
        return self.loader.get(idx)

    def pre(self, inp):
        """Run preprocessing
        Args:
            inp: input gotten from "getInput" or a reference to an input

        Returns:
            refDeps: A value suitable for passing to pool.run()'s refDeps argument
            preOutRef: A reference to the output of pre
        """
        preConf, preOutRef = _pre.remote(inp)
        return [preConf], preOutRef

    def getKaasReq(self, inputRef):
        """Generate a request for resnet suitable for passing to a pool of kaas
        workers.

        Args:
            inputRef: A reference to the input to the model. This is the
                "preOutRef" (second output) of pre()
        Returns:
            req: Value suitable for passing to the kaas executor worker
                in a pool.
        """

        packedReq = self.model.run(self.constRefs + [inputRef])
        reqRef = ray.put(packedReq)

        return reqRef

    def run(self, inp, pool, clientID):
        """Run resnet end-to-end in pool for clientID and return the pool's output.
        Arguments:
            inp:  Input value or reference to input value (gotten from getInput())
            pool: The kaas.pool to run this in. ClientID must already be
                  registered with the pool. For kaas mode, it should use the
                  kaas.ray.invokerActor.For tvm mode, it should use TvmWorker.
            clientID: ClientID to use with the pool.

        Returns:
            raw pool output: ref(ref([ref(resnetOut0), ref(resnetOut1)]))
                Calling ray.get() on the first two references may block. The
                inner two references (in the list) are guaranteed to be
                available.
        """
        preConf, preOutRef = self.pre(inp)

        if self.mode == 'kaas':
            reqRef = self.getKaasReq(preOutRef)

            poolResRef = pool.run(clientID, 'invoke',
                                  num_returns=self.model.nOutRun, refDeps=preConf,
                                  args=[reqRef], kwargs={"clientID": clientID})
        elif self.mode == 'tvm':
            poolResRef = pool.run(clientID, 'run', num_returns=self.model.nOutRun,
                                  refDeps=preConf, args=[[preOutRef]])
        else:
            raise ValueError("Unrecognized mode: ", self.mode)

        return poolResRef

    def getOutput(self, runOut):
        """Extract the actual return value from run(). This call will block
        until the return value is ready.
        Args:
            runOut: The output of run()

        Returns:
            int representing the resnet prediction
        """
        kaasOutRefs = ray.get(ray.get(runOut))
        rawPrediction = ray.get(kaasOutRefs[0])
        return int.from_bytes(rawPrediction, sys.byteorder) - 1


if __name__ == "__main__":
    mode = 'tvm'
    ray.init()

    groupID = 'resnetTest'
    pool = kaas.pool.Pool(infbench.util.getNGpu(), policy=kaas.pool.policies.BALANCE)
    if mode == 'kaas':
        pool.registerGroup(groupID, kaas.ray.invokerActor)
    elif mode == 'tvm':
        pool.registerGroup(groupID, TvmWorker)
    else:
        raise ValueError("Unrecognized mode: ", mode)

    resnetModel = ResnetHandle(pathlib.Path("resnetModel/resnet50"), mode)

    inp = resnetModel.getInput(0)
    outRefs = resnetModel.run(inp, pool, groupID)
    prediction = resnetModel.getOutput(outRefs)
    print(prediction)
