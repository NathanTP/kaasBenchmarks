CUTLASS_PATH ?= ../../../dependencies/cutlass

70FOUND:=$(shell lspci | grep "V100")

ifneq ($(70FOUND), )
	COMPUTE_CAP:=70
else
	COMPUTE_CAP:=35
endif

NVCC := nvcc 


all: getDims.so cutlass.cubin cutlassAdapters.so

getDims.so: getDims.cu
	${NVCC}  \
	-I$(CUTLASS_PATH)/include \
	-I$(CUTLASS_PATH)/g/examples/common \
	-I$(CUTLASS_PATH)/g/build/include \
	-I/usr/local/cuda/include \
	-I$(CUTLASS_PATH)/g/tools/util/include \
	-O0 -DNDEBUG -Xcompiler=-fPIC \
	-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_DEBUG_TRACE_LEVEL=0 \
	-Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing \
	-gencode=arch=compute_$(COMPUTE_CAP),code=sm_$(COMPUTE_CAP) -std=c++11 \
	-x cu -o $@ --shared $^ 

cutlassAdapters.so: cutlassAdapters.cu
	${NVCC}  \
	-I$(CUTLASS_PATH)/include \
	-I$(CUTLASS_PATH)/g/examples/common \
	-I$(CUTLASS_PATH)/g/build/include \
	-I/usr/local/cuda/include \
	-I$(CUTLASS_PATH)/g/tools/util/include \
	-O0 -DNDEBUG -Xcompiler=-fPIC \
	-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_DEBUG_TRACE_LEVEL=0 \
	-Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing \
	-gencode=arch=compute_$(COMPUTE_CAP),code=sm_$(COMPUTE_CAP) -std=c++11 \
	-x cu -o $@ --shared $^ 

cutlass.cubin: kern.cu
	${NVCC}  \
	-I$(CUTLASS_PATH)/include \
	-I$(CUTLASS_PATH)/g/examples/common \
	-I$(CUTLASS_PATH)/g/build/include \
	-I/usr/local/cuda/include \
	-I$(CUTLASS_PATH)/g/tools/util/include \
	-O0 -DNDEBUG -Xcompiler=-fPIC \
	--cubin \
	-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_DEBUG_TRACE_LEVEL=0 \
	-Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing \
	-gencode=arch=compute_$(COMPUTE_CAP),code=sm_$(COMPUTE_CAP) -std=c++11 \
	-x cu -c $^ -o $@ 

clean: 
	rm -f getDims.so cutlassAdapters.so cutlass.cubin
