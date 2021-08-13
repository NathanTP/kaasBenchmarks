from . import model
from . import dataset

import cv2
import numpy as np
import re
import sys


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


class resnet50Base(model.Model):
    noPost = True
    preMap = model.inputMap(inp=(0,))
    runMap = model.inputMap(pre=(0,))
    postMap = model.inputMap(run=(0,))
    nOutRun = 2
    nOutPre = 1
    nOutPost = nOutRun
    nConst = 0

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


class resnet50(model.tvmModel, resnet50Base):
    @staticmethod
    def getMlPerfCfg(gpuType, benchConfig):
        if gpuType == "Tesla K20c":
            maxQps = 28
            medianLatency = 0.07
        elif gpuType == "Tesla V100-SXM2-16GB":
            # Really wish I understood why this was so bad...
            maxQps = 6
            medianLatency = 0.14
        else:
            raise ValueError("Unrecoginzied GPU Type" + gpuType)

        settings = model.getDefaultMlPerfCfg(maxQps, medianLatency, benchConfig)

        return settings


class resnet50Kaas(model.kaasModel, resnet50Base):
    nConst = 108
    runMap = model.inputMap(const=range(108), pre=(0,))

    @staticmethod
    def getMlPerfCfg(gpuType, benchConfig):
        if gpuType == "Tesla K20c":
            maxQps = 31
            medianLatency = 0.07
        elif gpuType == "Tesla V100-SXM2-16GB":
            maxQps = 24
            medianLatency = 0.05
        else:
            raise ValueError("Unrecoginzied GPU Type" + gpuType)

        settings = model.getDefaultMlPerfCfg(maxQps, medianLatency, benchConfig)

        return settings


class imageNetLoader(dataset.loader):
    checkAvailable = True

    @property
    def ndata(self):
        try:
            return len(self.imageLabels)
        except AttributeError:
            raise RuntimeError("Accessed ndata before initialization")

    def __init__(self, dataDir):
        self.dataDir = dataDir / "fake_imagenet"

        # { imageID -> rawImage }
        # Loaded on-demand by preLoad
        self.images = {}

        # Dataset metadata, we only load actual data in preLoad (to avoid
        # loading all of a potentially large dataset)
        self.imagePaths = []
        self.imageLabels = []
        with open(self.dataDir / "val_map.txt", 'r') as f:
            nMissing = 0
            for imgRelPath in f:
                name, label = re.split(r"\s+", imgRelPath.strip())
                imgPath = self.dataDir / name

                if not imgPath.exists():
                    # OK to ignore missing images
                    nMissing += 1
                    continue

                self.imagePaths.append(imgPath)
                self.imageLabels.append(int(label))

    def get(self, idx):
        try:
            return (self.images[idx],)
        except KeyError as e:
            raise RuntimeError("Key {} not preloaded".format(e.args)) from e

    def preLoad(self, idxs):
        for i in idxs:
            cImg = cv2.imread(str(self.imagePaths[i]))
            bImg = cv2.imencode('.jpg', cImg)[1]
            self.images[i] = bImg.tobytes()

    def unLoad(self, idxs):
        for i in idxs:
            self.images[i] = None

    def check(self, result, idx):
        if not (isinstance(result[0], bytes) or isinstance(result[0], np.ndarray)):
            raise RuntimeError("Result has wrong type: expect bytes or ndarray, got ", type(result[0]))

        # I don't know why it's -1, but it is
        return (int.from_bytes(result[0], sys.byteorder) - 1) == self.imageLabels[idx]
