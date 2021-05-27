import time
scriptStart = time.time()

import ray
import serve

@ray.remote(num_returns=2)
def pre(imgBuf):
    return serve.imagePreprocess(imgBuf)


@ray.remote(num_gpus=1)
def run(modelBuf, img):
    ex = serve.importModelBuffer(modelBuf)
    return serve.executeModel(ex, img)


@ray.remote
def post(imgPil, imgOut):
    return serve.imagePostProcess(imgPil, imgOut)


@ray.remote(num_gpus=1)
def runInline(modelBuf, img):
    """Run model with inlined pre and post-processing"""
    imgPil, imgNp = serve.imagePreprocess(img)
    ex = serve.importModelBuffer(modelBuf)
    imgOut = serve.executeModel(ex, imgNp)
    res = serve.imagePostProcess(imgPil, imgOut)
    return res 


# This is remote so that we can asynchronously submit many jobs and have them
# wait on results
@ray.remote
def submitJob(imgRef, modelRef, name, preInline=False):
    if preInline:
        pngBuf = runInline.remote(modelRef, imgRef)
    else:
        imgPil, imgNp = pre.remote(imgRef)
        imgOut = run.remote(modelRef, imgNp)
        pngBuf = post.remote(imgPil, imgOut)

    # Trigger a fetch but don't do anything with it, this is just a benchmark
    ray.get(pngBuf)
    print(name + ": DONE")
    return True


stats = {}
def main():
    ray.init()
    stats['scriptStart'] = time.time() - scriptStart

    start = time.time()
    imgRef = serve.getImage()

    with open("super_resolution.so", 'rb') as f:
        modelRef = f.read()
    stats['modelInit'] = time.time() - start

    dones = []

    start = time.time()
    coldRes = submitJob.remote(imgRef, modelRef, "cold", preInline=True)
    ray.get(coldRes)
    stats['cold'] = time.time() - start

    start = time.time()
    for i in range(10):
        dones.append(submitJob.remote(imgRef, modelRef, str(i), preInline=True))

    ray.wait(dones, num_returns=len(dones))
    stats['warm'] = (time.time() - start) / 10

    # imgPil, imgNp = pre.remote(imgBuf)
    # imgOut = run.remote(modelBuf, imgNp)
    # pngBuf = post.remote(imgPil, imgOut)
    #
    # with open("test.png", "wb") as f:
    #     f.write(ray.get(pngBuf))

    print(stats)
main()
