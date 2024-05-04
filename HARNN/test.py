import torch.nn as nn
import torch
from torch.functional import F
from model.backbone import Backbone
from torchvision.models import resnet50, ResNet50_Weights
resnet50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

resnet50.layer1.register_forward_hook(get_activation('layer1'))
x = torch.randn(1, 3,224,224)
output = resnet50(x)
print(activation['layer1'].shape)

