# -*- coding:utf-8 -*-
__author__ = 'Ruben'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from PIL import Image

def truncated_normal(size, mean=0, std=1):
    lower_bound = mean - 2 * std
    upper_bound = mean + 2 * std
    samples = torch.empty(size)
    idx = 0

    while idx < samples.numel():
        candidate = torch.normal(mean, std, size)
        mask = torch.logical_and(candidate >= lower_bound, candidate <= upper_bound)
        valid_indices = torch.nonzero(mask)
        num_valid = min(samples.numel() - idx, valid_indices.size(0))
        if num_valid > 0:
            samples.view(-1)[idx:idx+num_valid] = candidate[mask][:num_valid]
            idx += num_valid

    return samples

class LeafModel(nn.Modules):
    """A HARNN for image classification."""
    def __init__(self,feature_dim,fc_hidden_size,fc_layer_num, num_classes,freeze_backbone,dropout_keep_prob=0.5,device=None):
        super(LeafModel,self).__init__()
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        self.backbone = torch.nn.Sequential(*(list(self.resnet50.children())[:len(list(self.resnet50.children()))])).to(device)
        self.device = device
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        self.fc_layer_cl = FullyConnectedClassifier(input_size=feature_dim,fc_layer_num=fc_layer_num,fc_hidden_size=fc_hidden_size,dropout_keep_prob=dropout_keep_prob,num_classes=num_classes,device=device)
    
    def forward(self,x):
        feature_out = self.backbone(x)
        logits = self.fc_layer_cl(feature_out)
        return logits
        
class FullyConnectedClassifier(nn.Modules):
    """A Fully Connected Block"""
    def __init__(self,input_size,fc_hidden_size,fc_layer_num,num_classes,dropout_keep_prob,device=None):
        super(FullyConnectedBlock,self).__init__()
        self.fc_layers = []
        for i in range(fc_layer_num):
            if i == 0:
                self.fc_layers.append(FullyConnectedBlock(input_size=input_size,fc_hidden_size=fc_hidden_size,dropout_keep_prob=dropout_keep_prob,device=device))
            else:
                self.fc_layers.append(FullyConnectedBlock(input_size=fc_hidden_size,fc_hidden_size=fc_hidden_size,dropout_keep_prob=dropout_keep_prob,device=device))
        self.linear_layer_cl = nn.Linear(fc_hidden_size,num_classes).to(device)
        
    def forward(self,x):
        output_fc = x
        for i in range(self.fc_layer_num):
            output_fc = self.fc_layers[i](output_fc)
        logits = self.linear_layer_cl(output_fc)
        return F.sigmoid(logits)

class FullyConnectedBlock(nn.Modules):
    """A Fully Connected Block"""
    def __init__(self,input_size,fc_hidden_size,dropout_keep_prob,device=None):
        super(FullyConnectedBlock,self).__init__()
        self.linear_layer=nn.Linear(input_size,fc_hidden_size).to(device)
        self.activation_layer = nn.ReLU(fc_hidden_size).to(device)
        self.batch_norm_layer = nn.BatchNorm1d(fc_hidden_size).to(device)
        self.dropout_layer = nn.Dropout1d(p=dropout_keep_prob).to(device)
    
    def forward(self, x):
        linear = self.linear_layer(x)
        activation = self.activation_layer(linear)
        batch_norm = self.batch_norm_layer(activation)
        dropout = self.dropout_layer(batch_norm)
        return dropout
