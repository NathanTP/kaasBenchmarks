import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io
import re
import cv2
import abc
import json


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


class loader(abc.ABC):
    """Handle to a dataset, used for reading inputs. Does not pre or post
    process data at all. Data are typically bytes or bytearrays."""
    # number of individual items in the dataset (the max index you could "get")
    ndata = 0

    def preLoad(self, idxs):
        """Some datasets are too big to fit in RAM in their entirety, preload
        will load a subset of data based on the indexes in idxs"""
        pass

    def unLoad(self, idxs):
        """Free memory associated with idxs"""
        pass

    @abc.abstractmethod
    def get(self, idx):
        """Returns a single datum at idx"""
        pass


    @abc.abstractmethod
    def check(self, result, idx):
        """Check if the result of index idx is correct"""
        pass


class superResLoader(loader):
    ndata = 1
    checkAvailable = True 

    def __init__(self, dataDir):
        imgPath = dataDir / "superRes" / "cat.png"
        self.img = Image.open(imgPath).tobytes()

        with open(dataDir / 'superRes' / 'catSupered.png', 'rb') as f:
            self.imgRef = f.read()


    def get(self, idx):
        if idx != 0:
            raise ArgumentError("The superres dataset has only one datum""")

        return (self.img,)


    def check(self, result, idx):
        return result[0] == self.imgRef


class imageNetLoader(loader):
    checkAvailable = True 

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
        # I don't know why it's -1, but it is
        return (int.from_bytes(result[0], sys.byteorder) - 1) == self.imageLabels[idx]


class cocoLoader(loader):
    checkAvailable = False

    def __init__(self, dataDir):
        self.dataDir = dataDir / "coco"

        with open(self.dataDir / "annotations/instances_val2017.json", "r") as f:
            meta = json.load(f)

        images = {}
        for img in meta["images"]:
            images[img["id"]] = {"file_name": img["file_name"],
                               "height": img["height"],
                               "width": img["width"],
                               "bbox": [],
                               "category": [],
                               'raw': None}

        for a in meta["annotations"]:
            img = images.get(a["image_id"])
            if img is None:
                continue
            catagory_ids = a.get("category_id")
            img["category"].append(catagory_ids)
            img["bbox"].append(a.get("bbox"))

        # At first, these have only metadata, but preLoad() can fill in a 'raw'
        # field to have the actual binary data.
        self.images = [ i for i in images.values() ]
        self.ndata = len(self.images)


    def preLoad(self, idxs):
        for i in idxs:
            img = self.images[i]
            imgPath = self.dataDir / 'val2017' / img['file_name']
            cImg = cv2.imread(str(imgPath))
            bImg = cv2.imencode('.jpg', cImg)[1]
            img['raw'] = bImg.tobytes()


    def get(self, idx):
        return (self.images[idx]['raw'],)


    def unload(self):
        for img in self.images:
            del img['raw']


    def check(self, result, idx):
        raise NotImplementedError("Check()")
