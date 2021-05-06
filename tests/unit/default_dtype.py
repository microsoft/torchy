from testdriver import *

torch.set_default_tensor_type(torch.DoubleTensor)

x = torch.tensor((1.,2.,3.))
y = torch.tensor((4.,5.,6.))

w = x.add(y)
print(w)

def check(t):
  print(t.dtype == torch.double)
  print(str(t.device) == 'cpu')

check(x)
check(y)
check(w)
