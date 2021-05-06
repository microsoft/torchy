from testdriver import *

torch.set_default_tensor_type(torch.cuda.HalfTensor)

x = torch.tensor(((3.,2.), (4.,5.)))
y = torch.tensor(((5.,6.), (7.,1.)))

w = x.add(y)
print(w)

def check(t):
  print(t.dtype == torch.float16)
  print(str(t.device).startswith('cuda'))

check(x)
check(y)
check(w)
