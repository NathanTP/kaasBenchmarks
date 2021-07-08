from . import model

import cv2
import numpy as np


def _centerCrop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img


def _resizeWithAspectratio(img, out_height, out_width, scale=87.5, inter_pol=cv2.INTER_LINEAR):
    height, width, _ = img.shape
    new_height = int(100. * out_height / scale)
    new_width = int(100. * out_width / scale)
    if height > width:
        w = new_width
        h = int(new_height * height / width)
    else:
        h = new_height
        w = int(new_width * width / height)
    img = cv2.resize(img, (w, h), interpolation=inter_pol)
    return img


class resnet50(model.tvmModel):
    noPost = True
    preMap = model.inputMap(inp=(0,))
    runMap = model.inputMap(pre=(0,))
    postMap = model.inputMap(run=(0,))
    nOutRun = 2
    nOutPre = 1
    nOutPost = nOutRun

    @staticmethod
    def pre(imgBuf):
        imgBuf = imgBuf[0]
        img = cv2.imdecode(np.frombuffer(imgBuf, dtype=np.uint8), flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        output_height, output_width, _ = [224, 224, 3]

        cv2_interpol = cv2.INTER_AREA
        img = _resizeWithAspectratio(img, output_height, output_width, inter_pol=cv2_interpol)
        img = _centerCrop(img, output_height, output_width)
        img = np.asarray(img, dtype='float32')

        # normalize image
        means = np.array([123.68, 116.78, 103.94], dtype=np.float32)
        img -= means

        img = img.transpose([2, 0, 1])

        return (img.tobytes(),)

    @staticmethod
    def post(label):
        raise AttributeError("resnet50 has no post-processing")

    @staticmethod
    def getMlPerfCfg(testing=False):
        settings = model.getDefaultMlPerfCfg()

        settings.server_target_qps = 3
        # if testing:
        #     settings.server_target_latency_ns = 1000
        # else:
        #     settings.server_target_latency_ns = 50000000

        return settings
