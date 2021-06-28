#!/bin/bash

PYTORCH=../pytorch
CFLAGS="-Wall -O3 -march=native"
LIB_DIR=`realpath $PYTORCH/build/lib.*/torch/lib`

for f in `ls scripts/*.cpp` ; do
  base=`basename $f .cpp`
  g++ -o $base $f $CFLAGS -I$PYTORCH/build/aten/src -I$PYTORCH/aten/src -I$PYTORCH -I$PYTORCH/build -L$LIB_DIR -Wl,-rpath=$LIB_DIR -lc10 -ltorch -ltorch_cpu
done
