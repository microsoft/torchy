import torch

x = torch.tensor(((3.,2.), (4.,5.)))

w = torch.tensor(((3.,2.), (4.,5.))).add_(y)
z = w.add(w)
w.add_(y)
print(w.mul(z))
