import torch
import torchvision.models as models

alexnet = models.alexnet(pretrained=False)
print(next(alexnet.parameters()).device)
x = alexnet
alexnet = alexnet.cuda()
print(next(x.parameters()).device)
print(next(alexnet.parameters()).device)
