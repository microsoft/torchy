from testdriver import *

x = torch.zeros([1,2])
y = torch.ones_like(x)
z = torch.ones([2,1])

x.add_(y)
w = x.t_()
w.add_(z)

print(w)
print(x)
