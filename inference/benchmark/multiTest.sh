#!/bin/bash

./benchmark.py -b client -n "client0" -t nshot --numRun=32 -m "superRes" --scale=0.5 &
./benchmark.py -b client -n "client1" -t nshot --numRun=32 -m "superRes" --scale=0.5 &
./benchmark.py -b client -n "client2" -t nshot --numRun=32 -m "superRes" --scale=0.5 &
./benchmark.py -b client -n "client3" -t nshot --numRun=32 -m "superRes" --scale=0.5
