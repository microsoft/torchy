<?php
// Copyright (c) 2021-present The Torchy Authors.
// Distributed under the MIT license that can be found in the LICENSE file.

$txt = <<< EOF
# Copyright (c) 2021-present The Torchy Authors.
# Distributed under the MIT license that can be found in the LICENSE file.

EOF;

process('types.txt', 'TYPES', 'type_inference', 'EQ_FIRST');
process('shapes.txt', 'SHAPES', 'shape_inference', null);
process('strides.txt', 'STRIDES', 'strides_inference', null);


function process($file, $title, $var, $default) {
  global $txt;

  $file = file_exists($file) ? $file : "../$file";

  foreach (file($file) as $line) {
    if (!trim($line) ||
        strstr($line, ' -> ') ||
        strstr($line, 'NON_STANDARD') ||
        strstr($line, 'NO_SAMPLES') ||
        strstr($line, 'MKL ERROR'))
      continue;
    preg_match('/([^:]+): (.*)/S', $line, $m);
    $data[$m[2]][] = $m[1];
  }

  ksort($data);

  $fns_txt = [];

  foreach ($data as $ty => $fns) {
    if ($default && $ty === $default)
      continue;

    foreach ($fns as $fn) {
      $fns_txt[] = "  '$fn': '$ty',";
    }
  }

  sort($fns_txt);
  $txt .= "\n$var = {\n";
  $txt .= implode("\n", $fns_txt);
  $txt .= "\n}\n";

  echo $title, "\n====\n";
  foreach ($data as $ty => $fns) {
    echo str_pad("$ty:", 20), sizeof($fns), "\n";
  }
  
  echo "\nNumber: ", sizeof($data), "\n\n";
}

file_put_contents('typings_data.py', $txt);
