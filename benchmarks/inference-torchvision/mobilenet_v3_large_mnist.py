from testdriver import *
import torchvision

model = torchvision.models.mobilenet_v3_large(pretrained=True)
if cuda:
  model.cuda()

def transform(img):
  t = torchvision.transforms.ToTensor()(img)
  # copy grayscale channel to 3 channels as this model expects RGB
  return t.repeat(3, 1, 1)

data = torchvision.datasets.MNIST('/tmp/torchy_data',
                                  train=False,
                                  download=True,
                                  transform=transform)

loader = torch.utils.data.DataLoader(data, batch_size=1)

model.eval()

for i, (data, label) in enumerate(loader):
  if i == 2000:
    break

  with torch.no_grad():
    if cuda:
      data = data.to('cuda')
    out = model(data)
    print(int(torch.argmax(out, dim=1)))
