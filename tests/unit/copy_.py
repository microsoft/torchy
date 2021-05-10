from testdriver import *

x = torch.tensor(((3.,2.), (4.,5.)))
y = torch.empty_like(x).copy_(x)

w = x.add(y)
y.add_(x)
z = x.add(x)

print(w)
print(z)
