# Copyright (c) 2021-present The Torchy Authors.
# Distributed under the MIT license that can be found in the LICENSE file.

import lit.TestRunner
import lit.util
from .base import TestFormat
import os, signal, subprocess, tempfile


def executeCommand(command):
  p = subprocess.Popen(command,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE,
                       env=os.environ)
  out,err = p.communicate()
  exit_code = p.wait()

  # Detect Ctrl-C in subprocess.
  if exit_code == -signal.SIGINT:
    raise KeyboardInterrupt

  # Ensure the resulting output is always of string type.
  try:
    out = str(out.decode('ascii'))
  except:
    out = str(out)
  try:
    err = str(err.decode('ascii'))
  except:
    err = str(err)
  return out, err, exit_code


class TorchyTest(TestFormat):
  def getTestsInDirectory(self, testSuite, path_in_suite, litConfig,
                          localConfig):
    source_path = testSuite.getSourcePath(path_in_suite)
    if 'lit' in source_path:
      return
    for filename in os.listdir(source_path):
      if filename.endswith('.py') and filename != 'lit.cfg.py':
        yield lit.Test.Test(testSuite, path_in_suite + (filename,), localConfig)


  def execute(self, test, litConfig):
    test = test.getSourcePath()

    out, err, exit_code = executeCommand(['python', test])
    if err or exit_code != 0:
      return lit.Test.FAIL, err + f'\nexit code: {exit_code}'

    out_torchy, err, exit_code = executeCommand(['python', test, '--torchy'])
    if exit_code != 0:
      return lit.Test.FAIL, err + f'\nexit code: {exit_code}'

    if out != out_torchy:
      return lit.Test.FAIL, f'{out}\nvs\n{out_torchy}'

    return lit.Test.PASS, ''
