UPGRADE PYTORCH
===============

To upgrade to a new PyTorch version, we need to add support for new operations.
Most of the times it's a matter of running a few scripts:

0) Patch PyTorch
Use `pytorch_disable_backtrace.patch` to disable backtrace generation when
an exception is generated. This gives a > 100x speedup when running the
infer_* programs.

1) Generate type inference rules
```
$ python scripts/infer_types.py
$ ./scripts/build.sh
$ ninja
```

This produces two files: types.txt & shapes.txt.
Look for new typings:
```
$ grep NON_STANDARD types.txt
```

Should be empty! If not, you need to add support for a new typing method
in the following files: infer_types.* and gen.py.
Note that shape inference is optional, while data-type (dtype) inference is not!

2) Regenerate C++ files in autogen folder
```
$ php scripts/gen_types_tables.php
$ python gen.py
```

3) Test if it compiles
```
$ python setup.py install
```

4) Run unit tests
```
$ ./run-tests.sh
```

5) Ship it! 🚀
