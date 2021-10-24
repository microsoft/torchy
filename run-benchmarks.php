<?php

define('NUM_RUNS', 32);

$files = [];
foreach (glob('benchmarks/*/*.py') as $file) {
  if (strstr($file, 'testdriver.py'))
    continue;
  $files[] = $file;
}

$tests = [
  ['CPU', '', ''],
  ['Cuda', '--cuda', ''],
  ['Torchy Int CPU', '--torchy', 'TORCHY_FORCE_INTERPRETER=1'],
  ['Torchy Int Cuda', '--torchy --cuda', 'TORCHY_FORCE_INTERPRETER=1'],
  ['Torchy TS CPU', '--torchy', ''],
  ['Torchy TS Cuda', '--torchy --cuda', ''],
];

// shuffle test runs to try to give a fair chance to all tests around vm load
$torun = [];
foreach ($files as $file) {
  foreach ($tests as $test) {
    for ($i = 0; $i < NUM_RUNS; ++$i) {
      $torun[] = [$file, $test[0], $test[1], $test[2]];
    }
  }
}

shuffle($torun);
echo "Running ", sizeof($torun), " tests\n";

$results = [];
$outputs = [];

foreach ($torun as $test) {
  run($test[0], $test[1], $test[2], $test[3]);
}


function test_name($file) {
  return substr(basename($file), 0, -3);
}

function run($file, $test, $args, $env) {
  global $outputs, $results;

  $name = test_name($file);
  $out = `$env /usr/bin/time -o time python $file $args 2> /dev/null`;
  if (empty($outputs[$name]))  {
    $outputs[$name] = $out;
  } elseif ($out !== $outputs[$name]) {
    echo "BUG: Output doesn't match for $name\n";
  }

  if (!preg_match('/(\d+):([0-9.]+)elapsed/S', file_get_contents('time'), $m))
    die("No time information!");
  $results[$name][$test][] = $m[1] * 60 + $m[2];
  unlink('time');
}

// table header
echo 'Benchmark';
foreach ($tests as $test) {
  echo ",$test[0]";
}
echo "\n";

// data
foreach ($files as $file) {
  $name = test_name($file);
  echo $name;
  foreach ($tests as $test) {
    echo ",", min($results[$name][$test[0]]);
  }
  echo "\n";
}
