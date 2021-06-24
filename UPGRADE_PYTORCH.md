UPGRADE PYTORCH
===============

To upgrade to a new PyTorch version, we need to add support for new operations.
Most of the times it's a matter of running a few scripts:

1) Generate type inference rules
$ python scripts/infer_types.py
$ ./scripts/build.sh
$ ninja

This produces a types.txt file.
Look for new typings:
$ grep NON_STANDARD types.txt

Should be empty! If not, you need to add support for a new typing method
in the following files: infer_types.* and gen.py.

2) Regenerate C++ files in autogen folder
$ php scripts/gen_types_tables.php
$ python gen.py

3) Test if it compiles
$ python setup.py install

4) Run unit tests
$ ./run-tests.sh

5) Ship it! 🚀
