#!/bin/bash

PYTORCH=../pytorch
LIB_DIR=`echo $PYTORCH/build/lib.*/torch/lib`

for f in `ls scripts/*.cpp` ; do
  base=`basename $f .cpp`
  g++ -o $base $f -Wall -I$PYTORCH/build/aten/src -I$PYTORCH/aten/src -I$PYTORCH -I$PYTORCH/build -O2 -L$LIB_DIR -Wl,-rpath=$LIB_DIR -lc10 -ltorch -ltorch_cpu
done
