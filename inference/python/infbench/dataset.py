from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io
import re


class processor():
    """Container for transformations over a dataset. Doesn't hold any complex
    refernces, it's really just a namespace.
    
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

    def pre(self, dat):
        pass

    def post(self, dat):
        pass


class loader():
    """Handle to a dataset, used for reading inputs. Does not pre or post
    process data at all. Data are typically bytes or bytearrays."""
    # number of individual items in the dataset (the max index you could "get")
    ndata = 0

    def preLoad(self, idxs):
        """Some datasets are too big to fit in RAM in their entirety, preload
        will load a subset of data based on the indexes in idxs"""
        pass

    def unload(self, idxs):
        """Free memory associated with idxs"""
        pass

    def get(self, idx):
        """Returns a single datum at idx"""
        pass


    def check(self, result, idx):
        """Check if the result of index idx is correct"""
        pass


class superResLoader(loader):
    ndata = 1

    def __init__(self, dataDir):
        imgPath = dataDir / "superRes" / "cat.png"
        self.img = Image.open(imgPath).tobytes()

        with open(dataDir / 'superRes' / 'catSupered.png', 'rb') as f:
            self.imgRef = f.read()

    def get(self, idx):
        if idx != 0:
            raise ArgumentError("The superres dataset has only one datum""")

        return self.img


    def check(self, result, idx):
        return result == self.imgRef


class imageNetLoader(loader):

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

        self.ndata = len(self.imageLabels)


    def get(self, idx):
        #XXX
        print("Getting ", idx)
        try:
            return self.images[idx]
        except KeyError as e:
            raise RuntimeError("Key {} not preloaded".format(e.args)) from e


    def preLoad(self, idxs):
        print("Asked to preload: ", idxs)
        for i in idxs:
            #XXX
            print("Preloading ", i)
            with open(self.imagePaths[i], 'rb') as f:
                self.images[i] = f.read() 


    def unload(self):
        self.images = {}


    def check(self, result, idx):
        return int(result) == self.imageLabels[idx]
