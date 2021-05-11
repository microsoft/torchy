from testdriver import *

x = torch.tensor(((3.,2.), (7.,9.)))
y = x.add(x)

w = y.clone()
y = None

k = x.add(w)

print(k)
print(x)
