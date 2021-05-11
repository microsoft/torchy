from testdriver import *

x = torch.tensor(((3.,2.), (7.,9.)))
y = x.detach_()

y.add_(x)

print(x)
print(y)
