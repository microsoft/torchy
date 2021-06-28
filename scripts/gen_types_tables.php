<?php
// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

$file = file_exists('types.txt') ? 'types.txt' : '../types.txt';

foreach (file($file) as $line) {
  if (strstr($line, ' -> ') || !trim($line))
    continue;
  preg_match('/([^:]+): (.*)/S', $line, $m);
  $data[$m[2]][] = $m[1];
}

ksort($data);

$txt = <<< EOF
# Copyright (c) 2021-present The Torchy Authors.
# Distributed under the MIT license that can be found in the LICENSE file.

type_inference = {

EOF;

$fns_txt = [];

foreach ($data as $ty => $fns) {
  // make eq first the default
  if ($ty === 'EQ_FIRST' || $ty === 'NO_SAMPLES' || $ty === 'NON_STANDARD:')
    continue;

  foreach ($fns as $fn) {
    $fns_txt[] = "  '$fn': '$ty',";
  }
}

sort($fns_txt);
$txt .= implode("\n", $fns_txt);

$txt .= "\n}\n";

file_put_contents('special_fns.py', $txt);

foreach ($data as $ty => $fns) {
  echo str_pad("$ty:", 20), sizeof($fns), "\n";
}

echo "\nNum Types: ", sizeof($data), "\n";
