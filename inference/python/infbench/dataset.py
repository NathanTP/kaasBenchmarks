from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io


class dataset():
    """Represents a particular dataset. All 'dat' arguments and return values
    are bytes unless otherwise specified.
    
    Methods may return multiple values, some or all of these values may be
    needed by subsequent steps. To track how things plug together, every
    dataset must have self.preMap and self.postMap fields. These are maps that
    list the indices of the previous steps that they want to consume (in
    order):
        nOutputPre: Number of return values of pre function
        nOutputPost: Number of return values of post function
        preMap:  tuple of indices of output from get
        postMap: Which outputs to take from pre, the model may only output one
                 value and that will be the last input given.
    """

    def get(self, idx):
        """Returns a single datum at idx"""
        pass

    def pre(self, dat):
        pass

    def post(self, dat):
        pass


class superResLoader():
    def __init__(self, dataDir):
        imgPath = dataDir / "superRes" / "cat.png"
        self.img = Image.open(imgPath).tobytes()


    def get(self, idx):
        if idx != 0:
            raise ArgumentError("The superres dataset has only one datum""")

        return self.img


class superResProcessor(dataset):
    nOutputPre = 2
    nOutputPost = 1
    preMap = (0,)
    postMap = (0,)
    inShape = (1, 1, 224, 224) 
    outShape = (1, 1, 672, 672)


    def pre(self, data):
        raw = data[0]
        # mode and size were manually read from the png (used Image.open and then
        # inspected the img.mode and img.size attributes). We're gonna just go with
        # this for now.
        img = Image.frombytes("RGB", (256,256), raw)

        imgProcessed = img.resize((224,224)).convert("YCbCr")
        img_y, img_cb, img_cr = imgProcessed.split()
        imgNp = (np.array(img_y)[np.newaxis, np.newaxis, :, :]).astype("float32")

        return (imgProcessed.tobytes(), imgNp.tobytes())


    def post(self, dat):
        imgPilRaw, imgRetRaw = dat

        retNp = np.frombuffer(imgRetRaw, dtype=np.float32)
        retNp.shape = self.outShape
        retNp = np.uint8((retNp[0, 0]).clip(0, 255))

        imgPil = Image.frombytes("YCbCr", (224,224), imgPilRaw)

        img_y, img_cb, img_cr = imgPil.split()
        out_y = Image.fromarray(retNp, mode="L")
        out_cb = img_cb.resize(out_y.size, Image.BICUBIC)
        out_cr = img_cr.resize(out_y.size, Image.BICUBIC)
        result = Image.merge("YCbCr", [out_y, out_cb, out_cr]).convert("RGB")
        canvas = np.full((672, 672 * 2, 3), 255)
        canvas[0:224, 0:224, :] = np.asarray(imgPil)
        canvas[:, 672:, :] = np.asarray(result)
        plt.imshow(canvas.astype(np.uint8))

        # Encode and serialize image as png buffer
        with io.BytesIO() as f:
            plt.savefig(f, format="png")
            pngBuf = f.getvalue()

        return (pngBuf)

