# Copyright (c) 2021-present The Torchy Authors.
# Distributed under the MIT license that can be found in the LICENSE file.

import lit.TestRunner
import lit.util
from .base import TestFormat
import os, signal, subprocess, tempfile


def executeCommand(command, env={}):
  env.update(os.environ)
  p = subprocess.Popen(command,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE,
                       env=env)
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


def check(out, name, cmd, env={}):
  out_torchy, err, exit_code = executeCommand(cmd, env)
  if exit_code != 0:
    return lit.Test.FAIL, f'{name}\n{err}\nexit code: {exit_code}'

  if out != out_torchy:
    return lit.Test.FAIL, f'{name}\n{out}\nvs\n{out_torchy}'

  return lit.Test.PASS, ''


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

    if exit_code == 0x42 and 'UNSUPPORTED' in out:
      return lit.Test.UNSUPPORTED, out

    if err or exit_code != 0:
      return lit.Test.FAIL, err + f'\nexit code: {exit_code}'

    err,msg = check(out, 'Torchy TS', ['python', test, '--torchy'])
    if err != lit.Test.PASS:
      return err, msg

    err,msg = check(out, 'Torchy Interpreter', ['python', test, '--torchy'],
                    {'TORCHY_FORCE_INTERPRETER' : '1'})
    if err != lit.Test.PASS:
      return err, msg

    return lit.Test.PASS, ''
