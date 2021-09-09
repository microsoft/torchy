from testdriver import *
import copy

xs = [torch.nn.Linear(in_features=768, out_features=768, bias=True) for i in range(33)]
ys = [copy.deepcopy(x) for x in xs]

for y in ys:
  print(y)
