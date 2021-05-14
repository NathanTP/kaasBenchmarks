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


# This is remote so that we can asynchronously submit many jobs and have them
# wait on results
@ray.remote
def submitJob(imgRef, modelRef, name):
    imgPil, imgNp = pre.remote(imgRef)
    imgOut = run.remote(modelRef, imgNp)
    pngBuf = post.remote(imgPil, imgOut)

    # Trigger a fetch but don't do anything with it, this is just a benchmark
    ray.get(pngBuf)
    print(name + ": DONE")
    return True

def main():
    ray.init()
    imgRef = serve.getImage()

    with open("super_resolution.tvm.tar.so", 'rb') as f:
        modelRef = f.read()

    dones = []
    for i in range(10):
        dones.append(submitJob.remote(imgRef, modelRef, str(i)))

    ray.wait(dones, num_returns=len(dones))

    # imgPil, imgNp = pre.remote(imgBuf)
    # imgOut = run.remote(modelBuf, imgNp)
    # pngBuf = post.remote(imgPil, imgOut)
    #
    # with open("test.png", "wb") as f:
    #     f.write(ray.get(pngBuf))

main()
