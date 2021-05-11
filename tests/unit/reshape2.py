from testdriver import *

x = torch.ones(3)
y = torch.tensor(((5.,6.,1.)))

x.add_(y)

print(x.reshape(-1))
print(x)
