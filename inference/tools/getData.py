#!/usr/bin/env python3
import pathlib
import shutil
import wget
import subprocess as sp

dataDir = (pathlib.Path(__file__).parent / ".." / "data").resolve()
cacheDir = pathlib.Path("/nscratch/datasets")


def getData(name, urls):
    print("\nGetting ", name)
    outDir = dataDir / name
    if not outDir.exists():
        if (cacheDir / name).exists():
            print("Fetching data from /nscratch/datasets")
            shutil.copytree(cacheDir / name, outDir)
        else:
            print("Downloading data")
            outDir.mkdir(0o700)
            for url in urls:
                wget.download(url, str(outDir))
    else:
        print("Already available, skipping")


bertDir = dataDir / "bert"
getData("bert", ["https://github.com/rajpurkar/SQuAD-explorer/blob/master/dataset/dev-v1.1.json?raw=true"])

cocoDir = dataDir / "coco"
getData("coco", ["http://images.cocodataset.org/annotations/annotations_trainval2017.zip", "http://images.cocodataset.org/zips/val2017.zip"])
cocoZips = [cocoDir / "annotations_trainval2017.zip", cocoDir / "val2017.zip"]
if cocoZips[0].exists():
    for f in cocoZips:
        sp.run(["unzip", str(f)], cwd=cocoDir, check=True)
        f.unlink()

print("Getting fake imagenet")
imagenetDir = dataDir / "fake_imagenet"
if not imagenetDir.exists():
    sp.run(["./make_fake_imagenet.sh", str(dataDir)], check=True)

getData("superRes", "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true")

superInput = dataDir / "superRes" / "cat.png"
superReference = dataDir / "superRes" / "catSupered.png"
if not superReference.exists():
    sp.run(["python", "superResReference.py", str(superInput)], check=True)
