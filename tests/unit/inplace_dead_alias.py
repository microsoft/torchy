from testdriver import *

x = torch.zeros([1,2])
y = torch.ones([2])
x.storage() # force flush

w = x.view([2])
w.add_(y)
w = None

print(x)
