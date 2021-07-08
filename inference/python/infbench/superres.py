from . import model

from PIL import Image
import numpy as np
import io
import matplotlib.pyplot as plt


class superRes(model.tvmModel):
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
        canvas[0:224, 0:224, :] = np.asarray(imgPil)
        canvas[:, 672:, :] = np.asarray(result)

        with io.BytesIO() as f:
            plt.imsave(f, canvas.astype(np.uint8), format="png")
            pngBuf = f.getvalue()

        return (pngBuf,)

    @staticmethod
    def getMlPerfCfg(testing=False):
        """Return a configuration for mlperf_inference. If testing==True, run a
        potentially invalid configuration that will run fast. This should ease
        testing for correctness."""
        settings = model.getDefaultMlPerfCfg()

        if testing:
            # MLperf detects an unatainable SLO pretty fast
            settings.server_target_qps = 3
            settings.server_target_latency_ns = 1000
        else:
            # Set this to the lowest qps that any system should be able to get
            # (benchmarks might fiddle with it to get a real measurement).
            settings.server_target_qps = 3

            # This is arbitrary for superRes
            settings.server_target_latency_ns = 1000000000

        return settings
