# ------------------------------------------------------------------------------------------------------
# model_loader_vgg16_deepfake.py
# Description       : Module which loads trained deep detector model with VGG-16 architecture
# Author            : Nazneen Mansoor
# Date              : 03/07/2023
# -------------------------------------------------------------------------------------------------------


import torchvision.models as models
import torch
from torch import nn
import numpy as np
import cv2

features_blobs = []


def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())


def loadmodel(fn):
    model = models.vgg16(pretrained=False)

    class Flatten(nn.Module):
        def forward(self, input):
            return input.view(input.size(0), -1)


    model.fc = nn.Linear(2048, 1)

    model.load_state_dict(torch.load(r'C:\Users\nazne\Desktop\project work python nb\project work python nb\Thesis project\DF_pytorch_vgg16_15epochs_lr0.0001.pth'), strict=False)
    model.features[21].register_forward_hook(fn)
    model.eval()

    return model

