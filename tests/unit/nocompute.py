import torch

x = torch.tensor(((3.,2.), (4.,5.)))
y = torch.tensor(((5.,6.), (7.,1.)))

w = x.add(x)
z = x.add(y)
w = None
z.mul_(x)
z.mul_(y)

print(z)
