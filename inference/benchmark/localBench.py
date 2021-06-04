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
                    "dataProc" : infbench.dataset.superResProcessor,
                    "model" : infbench.model.superRes,
                    "modelPath" : config['modelDir'] / "super_resolution.onnx"
                }
            }

def oneShot(modelName):
    modelSpec = models[modelName]
    loader = modelSpec['loader'](config['dataDir'])
    dataProc = modelSpec['dataProc']()
    model = modelSpec['model'](modelSpec['modelPath'])

    inp = loader.get(0)

    preOut = dataProc.pre([inp])

    modOut = model.run(preOut[model.inpMap[0]])

    postInp = [ preOut[i] for i in dataProc.postMap ] + [modOut]
    postOut = dataProc.post(postInp)
    return postOut
