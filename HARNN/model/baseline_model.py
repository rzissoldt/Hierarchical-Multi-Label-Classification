import torch
import torch.nn as nn
from model.backbone import Backbone



class BaselineModel(nn.Module):
    """Baseline model"""
    def __init__(self, output_dim, args):
        super(BaselineModel, self).__init__()
        print(args)
        self.backbone = Backbone(global_average_pooling_active=True)
        self.nb_layers = args.num_layers
        self.feature_dim = args.feature_dim_backbone[0]
        self.hidden_dim = args.fc_dim
        self.args = args
        if args.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        fc = []
        batchnorm = []
        for i in range(self.nb_layers):
            if self.nb_layers == 1:
                fc.append(nn.Linear(self.feature_dim, output_dim))
            elif i == self.nb_layers-1:
                fc.append(nn.Linear(self.hidden_dim, output_dim))
            elif i == 0:
                fc.append(nn.Linear(self.feature_dim, self.hidden_dim))
                batchnorm.append(nn.BatchNorm1d(self.hidden_dim))
            else:
                fc.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                batchnorm.append(nn.BatchNorm1d(self.hidden_dim))
        self.fc = nn.ModuleList(fc)
        self.batchnorm = nn.ModuleList(batchnorm)
        self.drop = nn.Dropout(args.dropout_rate)
        
        
        self.sigmoid = nn.Sigmoid()
        if args.activation_func == 'tanh':
            self.f = nn.Tanh()
        else:
            self.f = nn.ReLU()
        
    def forward(self, x):
        feature_extractor_out = self.backbone(x)
        x = torch.squeeze(feature_extractor_out,dim=2)
        for i in range(self.nb_layers):
            if i == self.nb_layers-1:
                x = self.sigmoid(self.fc[i](x))
            else:
                x = self.f(self.fc[i](x))
                if self.args.is_batchnorm_active:
                    if x.shape[0] != 1:
                        x = self.batchnorm[i](x)
                x = self.drop(x)
        
        return x
        
    
class BaselineModelLoss(nn.Module):
    def __init__(self):
        super(BaselineModelLoss, self).__init__()
        self.criterion = nn.BCELoss()
    def forward(self,x):
        predictions, targets = x
        global_loss = self.criterion(predictions.to(dtype=torch.float64),targets)
        return global_loss  