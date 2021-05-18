from testdriver import *

x = torch.tensor(((3.,2.), (4.,5.)))
y = torch.tensor(((5.,6.), (7.,1.)))

w = torch.stack([x,y])
z = w.add(w)
print(z)
print(w)
