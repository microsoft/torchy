#!/bin/sh

for f in `ls *.dot` ; do
  base=`basename $f .dot`
  dot -Tsvg $f > $base.svg &
done

wait
