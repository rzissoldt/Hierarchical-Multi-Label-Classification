import torch
import torch.nn as nn
import torch.functional as F
from HARNN.model.backbone import Backbone, BackboneEmbedding
class HmcLMLP(nn.Module):
    """A ANN for image classification."""
    
    def __init__(self,feature_dim,backbone_fc_hidden_size,fc_hidden_size, num_classes_list, freeze_backbone,dropout_keep_prob=0.2,beta=0.2,device=None):
        super(HmcLMLP,self).__init__()
        self.backbone = Backbone()
        self.backbone_embedding = BackboneEmbedding(feature_dim=feature_dim,backbone_fc_hidden_size=backbone_fc_hidden_size,dropout_keep_prob=dropout_keep_prob)
        self.feature_dim = feature_dim
        self.fc_hidden_size = fc_hidden_size
        self.num_classes_list = num_classes_list
        self.beta = beta
        self.total_classes = sum(num_classes_list)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        
        self.fc_activation = nn.ReLU()
        self.score_activation = nn.Sigmoid()
        fc_layers = []
        score_layers = []
        batchnorm_layers = []
        for i in range(len(num_classes_list)):
            if i == 0:
                fc_layer = nn.Linear(backbone_fc_hidden_size,fc_hidden_size)
            else:
                fc_layer = nn.Linear(num_classes_list[i-1]+backbone_fc_hidden_size,fc_hidden_size)
            score_layer = nn.Linear(fc_hidden_size,num_classes_list[i])
            batchnorm_layer = nn.BatchNorm1d(num_features=fc_hidden_size)
            fc_layers.append(fc_layer)
            score_layers.append(score_layer)
            batchnorm_layers.append(batchnorm_layer)        
        self.fc_layers = nn.ModuleList(fc_layers)
        self.score_layers = nn.ModuleList(score_layers)
        self.batchnorm_layers = nn.ModuleList(batchnorm_layers)
        self.dropout = nn.Dropout1d(p=dropout_keep_prob)
        
    def forward(self,x):
        input, level = x
        feature_out = self.backbone(input)
        feature_out_embedded = torch.squeeze(self.backbone_embedding(feature_out),dim=2)
        logits_list = []
        scores_list = []
        for i in range(level+1):
            if i == 0:
                fc = self.fc_layers[i](feature_out_embedded)
            else:
                input = torch.cat([scores_list[i-1],feature_out_embedded],dim=1)
                fc = self.fc_layers[i](input)
            fc_out = self.fc_activation(fc)
            if fc_out.shape[0] != 1:
                fc_out = self.batchnorm_layers[i](fc_out)
            fc_out_drop = self.dropout(fc_out)
            logits = self.score_layers[i](fc_out_drop)
            scores = self.score_activation(logits)
            logits_list.append(logits_list)
            scores_list.append(scores)
        
        return scores, scores_list
    
    
            
class HmcLMLPLoss(nn.Module):
    def __init__(self,l2_lambda,device=None):
        super(HmcLMLPLoss, self).__init__()
        self.l2_lambda = l2_lambda
        self.device = device
        self.criterion = nn.BCELoss()
    
    def forward(self,x):
        def _l2_loss(model,l2_reg_lambda):
            """Calculation of the L2-Regularization loss."""
            l2_loss = torch.tensor(0.,dtype=torch.float32).to(self.device)
            for param in model.parameters():
                if param.requires_grad == True:
                    l2_loss += torch.norm(param,p=2)**2
            return torch.tensor(l2_loss*l2_reg_lambda,dtype=torch.float32)
        predictions, targets, model = x
        local_loss = self.criterion(predictions,targets)        
        l2_loss = _l2_loss(model=model,l2_reg_lambda=self.l2_lambda)
        loss = torch.sum(torch.stack([local_loss,l2_loss]))
        return loss,local_loss,l2_loss
    

def activate_learning_level(model,optimizer,level):
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.fc_layers[level].parameters():
        param.requires_grad = True
        
    for param in model.score_layers[level].parameters():
        param.requires_grad = True
    
    for param in model.batchnorm_layers[level].parameters():
        param.requires_grad = True
    # Update optimizer to include parameters of the newly unfrozen layers
    unfrozen_params = []
    for layer in [model.fc_layers[level], model.score_layers[level],model.batchnorm_layers[level]]:
        unfrozen_params += filter(lambda p: p.requires_grad, layer.parameters())

    optimizer.param_groups[0]['params'] += unfrozen_params

    return model, optimizer
    