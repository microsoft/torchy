from testdriver import *

x = torch.tensor(((1.,2.), (3.,4.)))
y = torch.tensor(((5.,6.), (7.,8.)))

x.add_(y)
print(x.shape)
x.add_(y)

print(x)
