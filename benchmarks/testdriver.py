import torch
import sys

torch.manual_seed(0)

cuda = False

for arg in sys.argv[1:]:
  if arg == '--torchy':
    import torchy
    torchy.enable()
  elif arg == '--cuda':
    cuda = True
    if not torch.cuda.is_available():
      print('UNSUPPORTED: CUDA is not available')
      exit(0x42)
  else:
    print(f'Unknown arg: {arg}')
