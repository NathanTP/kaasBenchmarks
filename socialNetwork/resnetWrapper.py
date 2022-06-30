import infbench
from infbench import resnet50
import kaas

import pathlib
import ray
import sys


@kaas.pool.remote_with_confirmation()
def _pre(args):
    preProcessed = resnet50.resnet50Kaas.pre(args)
    return preProcessed[0]


class ResnetHandle():
    def __init__(self, modelDir):
        self.loader = resnet50.imageNetLoader(modelDir)
        self.loader.preLoad(range(self.loader.ndata))

        constants = resnet50.resnet50Kaas.getConstants(modelDir)
        self.constRefs = [ray.put(const) for const in constants]

        self.model = resnet50.resnet50Kaas(modelDir / 'resnet50_model.yaml',
                                           self.constRefs, backend='ray')

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

    def getReq(self, inputRef):
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
                  registered with the pool with the kaas.ray.invokerActor.
            clientID: ClientID to use with the pool.

        Returns:
            raw pool output: ref(ref([ref(resnetOut0), ref(resnetOut1)]))
                Calling ray.get() on the first two references may block. The
                inner two references (in the list) are guaranteed to be
                available.
        """
        preConf, preOutRef = self.pre(inp)

        reqRef = self.getReq(preOutRef)

        poolResRef = pool.run(clientID, 'invoke',
                              num_returns=self.model.nOutRun, refDeps=preConf,
                              args=[reqRef], kwargs={"clientID": clientID})

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
    ray.init()

    groupID = 'resnetTest'
    pool = kaas.pool.Pool(infbench.util.getNGpu(), policy=kaas.pool.policies.BALANCE)
    pool.registerGroup(groupID, kaas.ray.invokerActor)

    resnetModel = ResnetHandle(pathlib.Path("resnetModel/resnet50"))

    inp = resnetModel.getInput(0)
    outRefs = resnetModel.run(inp, pool, groupID)
    prediction = resnetModel.getOutput(outRefs)
    print(prediction)
