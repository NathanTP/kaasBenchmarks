import pathlib
from pprint import pprint

import infbench
import infbench.bert as bert
import numpy as np
import tvm

import localBench
import rayBench


dataDir = (pathlib.Path(__file__).parent / ".." / "data").resolve()
modelDir = (pathlib.Path(__file__).parent / ".." / "models").resolve()


def sanityCheck():
    """Basic check to make sure nothing is obviously broken. This is meant to
    be manually fiddled with to spot check stuff. It will run the superres
    model and write the output to test.png, it should be the superres output (a
    small cat next to a big cat in a figure)."""
    bench = localBench
    # bench = rayBench

    bench.configure({"dataDir": dataDir, "modelDir": modelDir})
    res = bench.nShot("superRes", 1)

    with open("test.png", "wb") as f:
        f.write(res[0][0])

    print("Sanity check didn't crash!")
    print("Output available at ./test.png")


def nshot(modelName):
    # bench = localBench
    bench = rayBench
    bench.configure({"dataDir": dataDir, "modelDir": modelDir})
    res = bench.nShot(modelName, 16, inline=True)


def runMlperf(modelName):
    testing = False
    inline = False
    backend = "ray"
    # backend = "local"

    if backend == 'ray':
        bench = rayBench
    elif backend == 'local':
        bench = localBench
    else:
        raise ValueError("Unrecognized backend: ", backend)

    bench.configure({"dataDir": dataDir, "modelDir": modelDir})

    print("Starting MLPerf Benchmark: ")
    print("\tModel: ", modelName)
    print("\tBackend: ", backend)
    print("\tTesting: ", testing)
    print("\tInline: ", inline)

    bench.mlperfBench(modelName, testing=testing, inline=inline)


def bertRawExample():
    # Examples are a combination of question and input text
    examples = bert.load(dataDir / 'bert' / 'bertInputs.json')[0:3]

    # Features are tokenized versions of the example. If the example input text
    # is too big, multiple features are generated for it based on a sliding
    # window over the input text.
    features = infbench.bert.featurize([examples[0]], modelDir / 'bert')

    exampleFeatures = features[0]
    example = examples[0]

    model, meta = infbench.model._loadSo(modelDir / 'bert' / 'bert.so')

    # The model only runs one feature at a time.
    feature = exampleFeatures[0]
    inputIds = np.array(feature.input_ids).astype(np.int64)[np.newaxis, :]
    inputMask = np.array(feature.input_mask).astype(np.int64)[np.newaxis, :]
    segmentIds = np.array(feature.segment_ids).astype(np.int64)[np.newaxis, :]
    model.set_input('input_ids', tvm.nd.array(inputIds))
    model.set_input('input_mask', tvm.nd.array(inputMask))
    model.set_input('segment_ids', tvm.nd.array(segmentIds))

    model.run()

    startLogits = model.get_output(0).numpy().astype(np.float32)[0].tolist()
    endLogits = model.get_output(1).numpy().astype(np.float32)[0].tolist()

    # Post-processing requires all features, even though the prediction used
    # only one feature.
    pred = bert.interpret(startLogits, endLogits, example, exampleFeatures)
    print("Final Prediction:")
    pprint(pred)


def main():
    sanityCheck()
    # nshot("resnet50")
    # nshot("ssdMobilenet")
    # nshot("mobilenet-ssd")
    # runMlperf("superRes")
    # runMlperf("resnet50")


main()
# bertRawExample()
