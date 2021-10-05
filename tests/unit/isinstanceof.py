from testdriver import *

x = torch.zeros([4, 3], dtype=torch.uint8)
print(isinstance(x, torch.ByteTensor))
print(x.dtype)

y = x.permute([0,-1])
print(isinstance(y, torch.ByteTensor))
print(y)
print(isinstance(y, torch.ByteTensor))
