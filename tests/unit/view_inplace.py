from testdriver import *

x = torch.tensor(((3.,8.), (9.,33.)))
y = torch.tensor((5.,6.,7.,1.))

x2 = x.view(4)
x2.add_(y)

print(x)
print(x2)
