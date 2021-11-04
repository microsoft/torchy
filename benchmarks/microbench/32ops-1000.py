from testdriver import *

device = 'cuda' if cuda else 'cpu'
size = [1000,1000]

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

for _ in range(50000):
  r = a.add(b).mul(c).div(d).add(e).sub(f).mul(g).div(h).add(i)
  r = r.sub(b).mul(c).div(d).add(e).sub(f).mul(g).div(h).add(i)
  r = r.sub(b).mul(c).div(d).add(e).sub(f).mul(g).div(h).add(i)
  r = r.sub(b).mul(c).div(d).add(e).sub(f).mul(g).div(h).add(i)
  r.storage()
