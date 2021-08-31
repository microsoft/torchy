from testdriver import *

x = torch.tensor(((3.,2.,5.)))
y = x.add(x).set_(torch.FloatStorage())

print(y)
