#!/bin/bash

for f in `ls tests/unit/*.py` ; do
  base=`basename $f`
  python "$f" --torchy &> "out/$base.txt" &
done

wait
