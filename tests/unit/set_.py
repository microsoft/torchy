from testdriver import *

x = torch.tensor(((3.,2.,5.)))
y = torch.empty([3])

y.set_(x.storage(), storage_offset=0, size=[2], stride=[0])

# force materialization
y.storage()

print(x)
print(y)
