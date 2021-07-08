from . import model

import cv2
import numpy as np
import io
import matplotlib.pyplot as plt
import mxnet

# Gluoncv throws out some stupid warning about having both mxnet and torch,
# have to go through this nonsense to suppress it.
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import gluoncv.data


cocoClassList = [u'person', u'bicycle', u'car', u'motorcycle', u'airplane',
                 u'bus', u'train', u'truck', u'boat', u'traffic light', u'fire hydrant',
                 u'stop sign', u'parking meter', u'bench', u'bird',
                 u'cat', u'dog', u'horse', u'sheep', u'cow', u'elephant',
                 u'bear', u'zebra', u'giraffe', u'backpack', u'umbrella',
                 u'handbag', u'tie', u'suitcase', u'frisbee', u'skis',
                 u'snowboard', u'sports ball', u'kite', u'baseball bat',
                 u'baseball glove', u'skateboard', u'surfboard', u'tennis racket',
                 u'bottle', u'wine', u'glass', u'cup', u'fork',
                 u'knife', u'spoon', u'bowl', u'banana', u'apple', u'sandwich',
                 u'orange', u'broccoli', u'carrot', u'hot dog', u'pizza',
                 u'donut', u'cake', u'chair', u'couch', u'potted plant',
                 u'bed', u'dining table', u'toilet', u'tv', u'laptop',
                 u'mouse', u'remote', u'keyboard', u'cell phone', u'microwave',
                 u'oven', u'toaster', u'sink', u'refrigerator', u'book',
                 u'clock', u'vase', u'scissors', u'teddy bear', u'hair drier',
                 u'toothbrush']


class ssdMobilenet(model.tvmModel):
    noPost = False
    preMap = model.inputMap(inp=(0,))
    runMap = model.inputMap(pre=(0,))
    postMap = model.inputMap(pre=(1,), run=(0, 1, 2))
    nOutPre = 2
    nOutRun = 3
    nOutPost = 1
    nConst = 0

    @staticmethod
    def pre(imgBuf):
        imgBuf = imgBuf[0]
        imgRaw = cv2.imdecode(np.frombuffer(imgBuf, dtype=np.uint8), flags=cv2.IMREAD_COLOR)
        imgRaw = cv2.cvtColor(imgRaw, cv2.COLOR_BGR2RGB)

        imgRaw = cv2.resize(imgRaw, (512, 512), interpolation=cv2.INTER_LINEAR)

        imgRaw = mxnet.nd.array(imgRaw).astype('uint8')
        imgMod, imgOrig = gluoncv.data.transforms.presets.ssd.transform_test(imgRaw, short=512)

        return (imgMod.asnumpy().tobytes(), imgOrig.tobytes())

    @staticmethod
    def post(modelOuts):
        imgOrig = np.frombuffer(modelOuts[0], dtype=np.uint8)
        cIDs = np.frombuffer(modelOuts[1], dtype=np.float32)
        scores = np.frombuffer(modelOuts[2], dtype=np.float32)
        bboxes = np.frombuffer(modelOuts[3], dtype=np.float32)
        imgOrig.shape = (512, 512, 3)
        cIDs.shape = (1, 100, 1)
        scores.shape = (1, 100, 1)
        bboxes.shape = (1, 100, 4)

        gluoncv.utils.viz.plot_bbox(
            imgOrig,
            bboxes[0],
            scores[0],
            cIDs[0],
            class_names=cocoClassList,
        )

        # Can't figure out how to save to buffer, easier to just trick pyplot
        with io.BytesIO() as f:
            plt.savefig(f, format="png")
            pngBuf = f.getvalue()

        return pngBuf

    @staticmethod
    def getMlPerfCfg(testing=False):
        settings = model.getDefaultMlPerfCfg()

        # XXX No idea right now
        if testing:
            settings.server_target_latency_ns = 1000
        else:
            settings.server_target_latency_ns = 1000000000

        return settings
