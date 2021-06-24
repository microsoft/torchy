<?php
// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

$file = file_exists('types.txt') ? 'types.txt' : '../types.txt';

foreach (file($file) as $line) {
  preg_match('/([^:]+): (.*)/S', $line, $m);
  $data[$m[2]][] = $m[1];
}

ksort($data);

$txt = <<< EOF
# Copyright (c) 2021-present The Torchy Authors.
# Distributed under the MIT license that can be found in the LICENSE file.

type_inferece = {

EOF;

foreach ($data as $ty => $fns) {
  // make eq first the default
  if ($ty === 'EQ_FIRST' || $ty === 'NO_SAMPLES')
    continue;

  foreach ($fns as $fn) {
    $txt .= "  '$fn': '$ty',\n";
  }
}
$txt .= "}\n";

file_put_contents('special_fns.py', $txt);

foreach ($data as $ty => $fns) {
  echo str_pad("$ty:", 20), sizeof($fns), "\n";
}

echo "\nNum Types: ", sizeof($data), "\n";
