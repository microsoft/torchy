from testdriver import *

device = 'cuda' if cuda else 'cpu'
size = [100,100]

a = torch.rand(size, device=device)
b = torch.rand(size, device=device)
c = torch.rand(size, device=device)
d = torch.rand(size, device=device)
e = torch.rand(size, device=device)
f = torch.rand(size, device=device)
g = torch.rand(size, device=device)
h = torch.rand(size, device=device)
i = torch.rand(size, device=device)
i.storage()

for _ in range(100000):
  r = a.add(b).mul(c).div(d).add(e).sub(f).mul(g).div(h).add(i)
  r.storage()
