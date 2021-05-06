from testdriver import *

dev = torch.device('cuda')
x = torch.tensor(((3.,2.), (4.,5.)), device=dev)
y = torch.tensor(((5.,6.), (7.,1.)), device=dev)

w = x.add(y)
print(w)
