#!/bin/bash
set -e

outDir=results/micro-$(date +%F-%H%M%S)
mkdir $outDir
echo -e "Output Dir: $outDir"

echo -e "\nRunning Actor nvprof cold"
CUDA_VISIBLE_DEVICES=0 nvprof -f --log-file $outDir/actorNvprofCold.csv --profile-from-start off --csv \
    ./benchmark.py -b local -e deepProf --force-cold -m testModelTvm

echo -e "\nRunning Actor nvprof warm"
CUDA_VISIBLE_DEVICES=0 nvprof -f --log-file $outDir/actorNvprofWarm.csv --profile-from-start off --csv \
    ./benchmark.py -b local -e deepProf -m testModelTvm

echo -e "\nRunning Actor Pipelined"
CUDA_VISIBLE_DEVICES=0 ./benchmark.py -b ray -e nshot -p exclusive -m testModelTvm
cp results.json $outDir/actorPipeline.json

echo -e "\nRunning Actor Inlined"
CUDA_VISIBLE_DEVICES=0 ./benchmark.py -b ray -e nshot -p exclusive -m testModelTvm --inline
cp results.json $outDir/actorInline.json

echo -e "\nRunning KaaS Pipelined"
CUDA_VISIBLE_DEVICES=0 ./benchmark.py -b ray -e nshot -p balance -m testModelKaas
cp results.json $outDir/kaasPipeline.json
