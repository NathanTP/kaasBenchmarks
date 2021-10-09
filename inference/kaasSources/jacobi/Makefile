70FOUND:=$(shell lspci | grep "V100")

ifneq ($(70FOUND), )
	COMPUTE_CAP:=70
else
	COMPUTE_CAP:=35
endif

NVCC := nvcc 

all: jacobi.ptx

jacobi.ptx: jacobi.cu
	$(NVCC) $^ --ptx -gencode arch=compute_$(COMPUTE_CAP),code=sm_$(COMPUTE_CAP) \
		-O3 -g -Xcompiler -Wall -Xcompiler -Wextra -std=c++14 -rdc=true \
        -I./Common \
	    -o $@

clean:
	rm -f *.cubin *.o *.ptx
