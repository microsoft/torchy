from testdriver import *

x = torch.ones(3)
y = torch.tensor(((5.,6.,1.)))

x.add_(y)
w = torch.add(x, y)
x = None

print(w)
print(y)
