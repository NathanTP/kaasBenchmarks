from . import model
from . import dataset

from PIL import Image
import numpy as np
import io
import matplotlib.pyplot as plt


class superResBase():
    preMap = model.inputMap(inp=(0,))
    runMap = model.inputMap(pre=(1,))
    postMap = model.inputMap(pre=(0,), run=(0,))
    nOutPre = 2
    nOutRun = 1
    nOutPost = 1

    noPost = False
    nConst = 0

    @staticmethod
    def pre(data):
        raw = data[0]
        # mode and size were manually read from the png (used Image.open and then
        # inspected the img.mode and img.size attributes). We're gonna just go with
        # this for now.
        img = Image.frombytes("RGB", (256, 256), raw)

        imgProcessed = img.resize((224, 224)).convert("YCbCr")
        img_y, img_cb, img_cr = imgProcessed.split()
        imgNp = (np.array(img_y)[np.newaxis, np.newaxis, :, :]).astype("float32")

        return (imgProcessed.tobytes(), imgNp.tobytes())

    @staticmethod
    def post(data):
        imgPilRaw = data[0]
        imgRetRaw = data[1]

        retNp = np.frombuffer(imgRetRaw, dtype=np.float32)
        retNp.shape = (1, 1, 672, 672)
        retNp = np.uint8((retNp[0, 0]).clip(0, 255))

        imgPil = Image.frombytes("YCbCr", (224, 224), imgPilRaw)

        img_y, img_cb, img_cr = imgPil.split()
        out_y = Image.fromarray(retNp, mode="L")
        out_cb = img_cb.resize(out_y.size, Image.BICUBIC)
        out_cr = img_cr.resize(out_y.size, Image.BICUBIC)
        result = Image.merge("YCbCr", [out_y, out_cb, out_cr]).convert("RGB")
        canvas = np.full((672, 672 * 2, 3), 255)
        canvas[0:224, 0:224, :] = np.asarray(imgPil.convert("RGB"))
        canvas[:, 672:, :] = np.asarray(result)

        with io.BytesIO() as f:
            plt.imsave(f, canvas.astype(np.uint8), format="png")
            pngBuf = f.getvalue()

        return (pngBuf,)

    @staticmethod
    def getMlPerfCfg(gpuType, testing=False):
        """Return a configuration for mlperf_inference. If testing==True, run a
        potentially invalid configuration that will run fast. This should ease
        testing for correctness."""
        settings = model.getDefaultMlPerfCfg()

        if gpuType == "Tesla K20c":
            settings.server_target_qps = 3

            settings.server_target_latency_ns = model.calculateLatencyTarget(0.320)
        else:
            raise ValueError("Unrecognized GPU Type: ", gpuType)

        return settings


class superResTvm(superResBase, model.tvmModel):
    pass


class superResKaas(superResBase, model.kaasModel):
    nConst = 8
    runMap = model.inputMap(const=(0, 1, 2, 3, 4, 5, 6, 7,), pre=(1,))


class superResLoader(dataset.loader):
    ndata = 1
    checkAvailable = True

    def __init__(self, dataDir):
        self.preLoaded = False
        imgPath = dataDir / "superRes" / "cat.png"
        self.img = Image.open(imgPath).tobytes()

        self.imgRef = Image.open(dataDir / 'superRes' / 'catSupered.png')

    def preLoad(self, idxs):
        self.preLoaded = True

    def unLoad(self, idxs):
        self.preLoaded = False
        pass

    def get(self, idx):
        if not self.preLoaded:
            raise RuntimeError("Called get before preloading")

        if idx != 0:
            raise ValueError("The superres dataset has only one datum""")

        return (self.img,)

    def check(self, result, idx):
        npIO = io.BytesIO(result[0])
        imgRes = Image.open(npIO)

        npRes = np.asarray(imgRes).astype('float32')
        npRef = np.asarray(self.imgRef).astype('float32')

        # SuperRes isn't completely deterministic, but so long as all the pixel
        # values are within 5 (out of 255) of eachother, it probably did the
        # right thing
        return np.allclose(npRes, npRef, atol=5)
