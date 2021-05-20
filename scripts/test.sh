#!/bin/bash

for f in `ls tests/unit/*.py` ; do
  base=`basename $f`
  echo $f
  python "$f" --torchy |& diff -u - "out/$base.txt"
done
