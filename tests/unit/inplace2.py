from testdriver import *

x = torch.tensor(((3.,2.), (4.,5.)))
y = torch.tensor(((5.,6.), (7.,1.)))

w = x.add(x)
x.add_(y)
w = torch.add(w, x)
print(w)
