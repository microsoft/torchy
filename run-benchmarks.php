<?php

define('NUM_RUNS', 9);

$files = glob('benchmarks/*/*.py');

$outputs = [];
$results = [];

$tests = [
  ['CPU', '', ''],
  ['Cuda', '--cuda', ''],
  ['Torchy Int CPU', '--torchy', 'TORCHY_FORCE_INTERPRETER=1'],
  ['Torchy Int Cuda', '--torchy --cuda', 'TORCHY_FORCE_INTERPRETER=1'],
  ['Torchy TS CPU', '--torchy', ''],
  ['Torchy TS Cuda', '--torchy --cuda', ''],
];

foreach ($files as $file) {
  if (strstr($file, 'testdriver.py'))
    continue;

  $name = substr(basename($file), 0, -3);
  echo "Running $name\n";

  $outputs = [];
  foreach ($tests as $test) {
    run($file, $test[0], $test[1], $test[2]);
  }

  foreach ($outputs as $out) {
    if ($out !== $outputs[0]) {
      die("BUG: Output doesn't match!");
    }
  }
}

function test_name($file) {
  return substr(basename($file), 0, -3);
}

function run($file, $test, $args, $env) {
  global $outputs, $results;

  $name = test_name($file);
  echo "Running $name on $test\n";

  $times = [];
  for ($i = 0; $i < NUM_RUNS; ++$i) {
    $outputs[] = `$env /usr/bin/time -o time python $file $args 2> /dev/null`;

    preg_match('/(\d+):([0-9.]+)elapsed/S', file_get_contents('time'), $m);
    $times[] = $m[1] * 60 + $m[2];
    unlink('time');
  }
  // median
  sort($times);
  $results[$name][$test] = $times[(int)(NUM_RUNS/2)];
}

// table header
foreach ($test as $test) {
  echo ",$test[0]";
}
echo "\n";

// data
foreach ($files as $file) {
  $name = test_name($file);
  echo $name;
  foreach ($tests as $test) {
    echo ",", $results[$name][$test[0]];
  }
  echo "\n";
}
