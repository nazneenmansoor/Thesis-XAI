# ------------------------------------------------------------------------------------------------------
# model_loader_resnet50_deepfake.py
# Description       : Module which loads trained deep detector model with ResNet-50 architecture
# Author            : Nazneen Mansoor
# Date              : 07/07/2023
# -------------------------------------------------------------------------------------------------------


import torchvision.models as models
import torch

from torch import nn

features_blobs = []

def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())


def loadmodel(fn):
    model = models.resnet50(pretrained=False)

    class Flatten(nn.Module):
        def forward(self, input):
            return input.view(input.size(0), -1)

    model.fc = nn.Linear(2048, 1)

    model.load_state_dict(torch.load(r'C:\Users\nazne\Desktop\project work python nb\project work python nb\Thesis project\DF_pytorch_model1_resnet50_15epochs_lr0.00001.pth'), strict=False)
    model.layer3[5].conv3.register_forward_hook(fn)
    model.eval()

    return model


