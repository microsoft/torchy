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
  elif arg == '--fuser-nnc':
    torch._C._jit_override_can_fuse_on_cpu(True)
    torch._C._jit_override_can_fuse_on_gpu(True)
  elif arg == '--nvfuser':
    #os.environ['PYTORCH_CUDA_FUSER_DISABLE_FALLBACK'] = '1'
    #os.environ['PYTORCH_CUDA_FUSER_DISABLE_FMA'] = '1'
    #os.environ['PYTORCH_CUDA_FUSER_JIT_OPT_LEVEL'] = '0'
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_nvfuser_enabled(True)
  else:
    print(f'Unknown arg: {arg}')
