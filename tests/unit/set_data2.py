from testdriver import *

x = torch.zeros([2])
y = torch.ones([2])
x = x.add(x)
y = y.add(x)
y.storage() # force flush

w = x.add(y)
w.data = x

print(x)
print(y)
print(w)
