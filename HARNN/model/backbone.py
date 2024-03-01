import torch
import torch.nn as nn
import torch.nn.functional as F

class BackboneEmbedding(nn.Module):
    """A Backbone for image feature extraction."""
    def __init__(self,feature_dim,backbone_fc_hidden_size,dropout_keep_prob):
        super(BackboneEmbedding,self).__init__()
        self.backbone_fc_layer = nn.Linear(in_features=feature_dim[0],out_features=backbone_fc_hidden_size)
        self.backbone_activation = nn.ReLU()
        self.backbone_batchnorm = nn.BatchNorm1d(backbone_fc_hidden_size)
        self.backbone_dropout =  nn.Dropout1d(p=dropout_keep_prob)
    
    def forward(self,x):
        fc_feature_out = self.backbone_fc_layer(x)
        fc_feature_out = self.backbone_activation(fc_feature_out)
        if fc_feature_out.shape[0] != 1:
            fc_feature_out = self.backbone_batchnorm(fc_feature_out)
        fc_feature_out = self.backbone_dropout(fc_feature_out)
        fc_feature_out = torch.unsqueeze(fc_feature_out,dim=2)
        return fc_feature_out
    
class Backbone(nn.Module):
    """A Backbone for image feature extraction."""
    def __init__(self):
        super(Backbone,self).__init__()
        resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        self.backbone_feature_ext = nn.Sequential(*(list(resnet50.children())[:len(list(resnet50.children()))-1]))
        
    def forward(self,x):
        feature_extractor_out = self.backbone_feature_ext(x)
        num_channels,spatial_dim1, spatial_dim2 = feature_extractor_out.shape[1:]
        feature_extractor_out = feature_extractor_out.view(-1, num_channels, spatial_dim1 * spatial_dim2)
        return feature_extractor_out