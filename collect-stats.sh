#!/bin/bash

for f in `ls benchmarks/inference*/*.py | grep -v testdriver` ; do
  base=`basename $f`
  python $f --torchy &> "stats/$base.txt" &
done

wait
