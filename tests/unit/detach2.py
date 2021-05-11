from testdriver import *

x = torch.tensor(((3.,2.), (7.,9.)))
y = x.clone().detach()
print(y)
