from testdriver import *

x = torch.tensor(((3.,2.), (4.,5.), (7.,9.)))
y = torch.tensor(((5.,6.), (7.,1.)))

x.resize_as_(y)
print(x.add(y))
