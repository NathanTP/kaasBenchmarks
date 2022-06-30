#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $SCRIPT_DIR

set +e
git submodule status | grep "^-" > /dev/null
subsInitialized=$?
if [ $subsInitialized == 0 ]; then
    git submodule update --init --recursive
fi
set -e

pushd dependencies/tvm
    mkdir -p build
    cp ../tvm_config.cmake build/config.cmake
    pushd build
        cmake ..
        make -j8
    popd

    pushd python
        pip install -e .
    popd
popd

pushd dependencies/kaas/python
    pip install -e .
popd

pushd dependencies/mlperf/loadgen
    pip install absl-py numpy
    CFLAGS="-std=c++14 -O3" python setup.py bdist_wheel
    pip install --force-reinstall dist/mlperf_loadgen-1.1-cp39-cp39-linux_x86_64.whl
popd

pushd inference
    pip install -r requirements.txt

    pushd python
        pip install -e .
    popd

    pushd tools
        ./getData.py
        ./getModels.py
    popd
popd
