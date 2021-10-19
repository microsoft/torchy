from testdriver import *

x = torch.zeros([2])
y = torch.ones([2])
x = x.add(x)
y = y.add(x)
y.storage() # force flush

w = x.add(y)
x.data = y

print(x)
print(y)
print(w)
