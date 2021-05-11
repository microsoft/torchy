from testdriver import *

x = torch.tensor(((3.,2.), (7.,9.)))
y = x.add(x)

w = y.clone()
w = None

k = y.add(y)

print(y)
print(x)
