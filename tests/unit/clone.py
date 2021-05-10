from testdriver import *

x = torch.tensor(((3.,2.), (7.,9.)))
y = x.clone().detach()

w = x.add(y)
y.add_(x)
z = x.add(x)

print(w)
print(z)
