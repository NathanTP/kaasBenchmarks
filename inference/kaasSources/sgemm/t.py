import libff as ff
import libff.kv
import libff.invoke
import libff.kaas as kaas
import libff.kaas.kaasFF

import numpy as np
import yaml

objStore = ff.kv.Local(copyObjs=False, serialize=False)
ctx = ff.invoke.RemoteCtx(None, objStore)
kaasHandle = kaas.kaasFF.getHandle('direct', ctx)

with open("sgemm_model.yaml", 'r') as f:
    rDict = yaml.safe_load(f)

req = kaas.kaasReq.fromDict(rDict)

shape = (128, 128)
inputA = np.full(shape, 10, dtype=np.float32)

const0 = np.zeros(shape, dtype=np.float32)
np.fill_diagonal(const0, 1)

const1 = np.zeros(shape, dtype=np.float32)
np.fill_diagonal(const1, 2)

const2 = np.zeros(shape, dtype=np.float32)
np.fill_diagonal(const2, 3)

#XXX
inputA += 1

ctx.kv.put("inputA", inputA)
ctx.kv.put("inputB", const0)
ctx.kv.put("intermediate0B", const1)
ctx.kv.put("outputB", const2)

kaasHandle.Invoke(req.toDict())

cRaw = ctx.kv.get('outputC')
cArr = np.frombuffer(cRaw, dtype=np.float32)
testRes = cArr.reshape(128, 128)

#XXX
print(testRes - 1)

expect = np.matmul(inputA, const0)
expect = np.matmul(expect, const1)
expect = np.matmul(expect, const2)

if np.allclose(expect, testRes, rtol=0.05):
    print("PASS")
else:
    print("FAIL")
    dist = np.linalg.norm(expect - testRes)
    print("Returned matrix doesn't look right: l2=", dist)
