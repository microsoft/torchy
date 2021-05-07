from testdriver import *

x = torch.ones(3)
y = torch.tensor(((5.,6.,1.)))

x.add_(y)
x.mul_(y)
y.add_(x)

print(x)
print(y)
