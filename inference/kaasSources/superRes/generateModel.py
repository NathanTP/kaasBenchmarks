import yaml
import sys
import pathlib
import pickle
import json
from superRes import createReq


cwd = pathlib.Path(__file__).parent.resolve()

def loadGraph():
    graph = open(cwd / "graph.json")
    return json.load(graph)

'''This method is useful because the intermediate buffers in multi-kernel nodes aren't present in the graph, so this code is needed in 2 separate locations. '''
def getInfo(buf, graph):
    name = buf.name
    index = int(name)
    dtype = graph["attrs"]["dltype"][1][index]
    shape = graph["attrs"]["shape"][1][index]
    return dtype, shape

def loadParams():
    path = cwd / "params.pkl"
    return pickle.load(open(path, 'rb')) 


def metaFromReq(req):
    graph = loadGraph()
    c = 0
    constants = []
    inputs = []
    outputs = []
    for kern in req.kernels:
        for buf in kern.inputs:
            if not buf.ephemeral:
                dtype, shape = getInfo(buf, graph)
                if buf.const:
                    c += 1
                    constants.append({"name": buf.name, "type": dtype, "shape": shape})
                else:
                    inputs.append({"name": buf.name, "type": dtype, "shape": shape})
        for buf in kern.outputs:
            if not buf.ephemeral:
                dtype,shape = getInfo(buf, graph)
                outputs.append({"name": buf.name, "type": dtype, "shape": shape})
    print(c)
    return {"constants": constants, "inputs": inputs, "outputs": outputs}


def getParams():
    params = loadParams()
    params_list = []
    for i in range(len(params.keys())):
        params_list.append(params["p" + str(i)])
    return params_list


if __name__ == "__main__":

    req = createReq()
    with open(cwd / "superRes_model.yaml", 'w') as f:
        yaml.safe_dump(req.toDict(), f)

    meta_data = metaFromReq(req)
    with open(cwd / "superRes_meta.yaml", 'w') as f:
        yaml.safe_dump(meta_data, f)

    params = getParams()
    with open(cwd / "superRes_params.pkl", 'wb') as f:
        pickle.dump(params, f)

    
