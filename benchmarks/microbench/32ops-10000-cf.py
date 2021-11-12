from testdriver import *

device = 'cuda' if cuda else 'cpu'
size = [10000,10000]

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

def fn(a, b, c, d, e, f, g, h, i, var : int):
  r = a.add(b).mul(c).div(d).add(e).sub(f).mul(g).div(h).add(i)
  if var == 0:
    r = r.sub(b).mul(c).div(d).add(e).sub(f).mul(g).div(h).add(i)
    r = r.sub(b).mul(c).div(d).add(e).sub(f).mul(g).div(h).add(i)
  else:
    r = r.sub(c).div(c).mul(d).sub(e).mul(f).div(g).add(h).add(i)
    r = r.sub(c).div(c).mul(d).sub(e).mul(f).div(g).add(h).add(i)
  r = r.sub(b).mul(c).div(d).add(e).sub(f).mul(g).div(h).add(i)
  return r

if torchscript:
  fn = torch.jit.script(fn)

for _ in range(500):
  fn(a, b, c, d, e, f, g, h, i, 0).storage()
  fn(a, b, c, d, e, f, g, h, i, 1).storage()
