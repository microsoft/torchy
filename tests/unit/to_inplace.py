from testdriver import *

x = torch.tensor(((3.,8.), (9.,33.)))
y = torch.tensor(((5.,6.), (7.,1.)))

x2 = x.to(torch.float, copy=False)
x2.add_(y)

print(x)
print(x2)
