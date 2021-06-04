import ray
import infbench

config = {}
models = {}

def configure(cfg):
    """Must include dataDir and modelDir fields (pathlib paths)"""
    global config
    global models

    config = cfg

    models = {
            "superRes" : {
                    "loader" : infbench.dataset.superResLoader,
                    "dataProcClass" : infbench.dataset.superResProcessor,
                    "modelClass" : infbench.model.superRes,
                    "modelPath" : config['modelDir'] / "super_resolution.onnx"
                }
            }


@ray.remote
def pre(modelName, data):
    dataProc = models[modelName]['dataProcClass']()
    return dataProc.pre(data)


@ray.remote(num_gpus=1)
def run(modelName, modelBuf, data):
    model = models[modelName]['modelClass'](modelBuf)
    return model.run(data)


@ray.remote
def post(modelName, dataRefs):
    dataProc = models[modelName]['dataProcClass']()
    data = [ ray.get(r) for r in dataRefs ]
    return dataProc.post(data)


def oneShot(modelName):
    """Single invocation of the model. This test assumes you have compiled the
    model at least once (the .so is available in the cache)."""
    ray.init()

    modelSpec = models[modelName]
    loader = modelSpec['loader'](config['dataDir'])
    modelBuf = infbench.model.loadModelBuffer(modelSpec['modelPath'])
    dataProc = modelSpec['dataProcClass']

    inp = loader.get(0)

    preOut = pre.options(num_returns=dataProc.nOutputPre).remote(modelName, [inp])

    modOut = run.options(num_returns=modelSpec['modelClass'].nOutput).remote(
                 modelName, modelBuf, preOut[modelSpec['modelClass'].inpMap[0]])

    postInp = [ preOut[i] for i in dataProc.postMap ] + [modOut]
    postOut = post.options(num_returns=dataProc.nOutputPost).remote(modelName, postInp)
    return ray.get(postOut)
