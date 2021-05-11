from testdriver import *

x = torch.tensor(((3.,2.), (7.,9.)))
y = x.detach()

y.add_(x)
x.mul_(y)

print(y)
print(x)
