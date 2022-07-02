70FOUND:=$(shell lspci | grep "V100")

ifneq ($(70FOUND), )
	COMPUTE_CAP:=70
else
	COMPUTE_CAP:=35
endif

NVCC := nvcc
NVCC_OPTS := -O3 -g -arch=compute_$(COMPUTE_CAP) -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -fPIC -m64 -std=c++14
LIBS :=  -lcuda
PYTHON := python

.PHONY: all

resnet50/resnet50.cubin: kerns/source.cu
	nvcc $^ --cubin -gencode arch=compute_$(COMPUTE_CAP),code=sm_$(COMPUTE_CAP) -O3 -g -Xcompiler -Wall -Xcompiler -Wextra -std=c++14 -o $@
	$(PYTHON) generateModel.py

%.o: %.cu
	$(NVCC) -c $< $(NVCC_OPTS)

clean:
	rm -f *.cubin *.o
