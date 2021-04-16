Torchy
======

A tracing JIT for PyTorch.
WIP; don't use.


Install
-------
$ python setup.py install


Run
---
Torchy shouldn't require any change beyond adding a call to `torchy.enable()`.
Example:

```
import torch
import torchy

torchy.enable()

x = torch.tensor(((3.,2.), (4.,5.)))
y = torch.tensor(((5.,6.), (7.,1.)))

w = x.add(x)
z = x.add(y)
w = None  # w not computed
print(z)
```
