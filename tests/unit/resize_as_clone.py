from testdriver import *

x = torch.tensor(((3.,2.), (4.,5.), (7.,9.)))
y = torch.tensor(((3.,2.), (1.,8.), (7.,1.)))
z = torch.tensor(((5.,6.), (7.,1.)))

x2 = x.clone()

x.resize_as_(z)
x.add_(z)

x2.add_(y)
y.mul_(x2)

print(x)
print(x2)
print(y)
print(z)
