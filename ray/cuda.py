import ray


# PyCUDA stuff
import os
import pycuda.driver as cuda
# import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

# Ray sets CUDA_VISIBLE_DEVICES but pycuda seems to re-number devices based on
# it, so we shouldn't use it ourselves. DevId=0 is just the first device that
# is available to us based on CUDA_VISIBLE_DEVICES
def runCuda(a, devId=0):
    cuda.init()
    
    cudaDev = cuda.Device(devId)
    print("Dev pcie: ", cudaDev.pci_bus_id())
    cudaCtx = cudaDev.make_context()


    mod = SourceModule("""
      __global__ void doublify(float *a)
      {
        int idx = threadIdx.x + threadIdx.y*4;
        a[idx] *= 2;
      }
      """)
    func = mod.get_function("doublify")

    a_gpu = cuda.mem_alloc(a.nbytes)
    cuda.memcpy_htod(a_gpu, a)

    func(a_gpu, block=(4,4,1))

    a_doubled = np.empty_like(a)
    cuda.memcpy_dtoh(a_doubled, a_gpu)

    cudaCtx.pop()

    return a_doubled


@ray.remote(num_gpus=1)
def runCudaRay(a, devId=0):
    return runCuda(a, devId)


def testRay():
    ray.init()

    a = np.random.randn(4,4)
    a = a.astype(np.float32)

    nCopy = 4
    futList = [ runCudaRay.remote(a) for i in range(nCopy) ]
    ray.get(futList[0])
    ray.get(futList[1])
    rList = ray.get(futList)

    rExpect = a*2
    for i,f in enumerate(rList):
        if np.array_equal(f, rExpect):
            print("Result {}: Passed")
        else:
            print("Result {}: Failed")


def testNoRay():
    a = np.random.randn(4,4)
    a = a.astype(np.float32)

    nCopy = 2
    rList = [ runCuda(a) for i in range(nCopy) ]

    print(type(rList[0]))
    rExpect = a*2
    for i,f in enumerate(rList):
        if np.array_equal(f, rExpect):
            print("Result {}: Passed")
        else:
            print("Result {}: Failed")


# testNoRay()
testRay()
