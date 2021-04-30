import torch

y = torch.tensor(((5.,6.), (7.,1.)))

w = torch.tensor(((3.,2.), (4.,5.))).add_(y)
z = w.add(w)
w.add_(y)
print(w.mul(z))
