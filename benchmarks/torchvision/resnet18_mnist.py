from testdriver import *
import torchvision

model = torchvision.models.resnet18(pretrained=True)

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
  if i == 1000:
    break

  with torch.no_grad():
    out = model(data)
    print(torch.argmax(out, dim=1))
