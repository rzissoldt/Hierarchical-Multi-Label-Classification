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

class TCCA(nn.Module):
    """TCCA Module"""
    def __init__(self,feature_dim,num_classes,attention_unit_size):
        super(TCCA, self).__init__()
        num_channels, spatial_dim = feature_dim
        self.W_s1 = nn.Parameter(truncated_normal(size=(attention_unit_size,spatial_dim),std=0.1))
        self.W_s2 = nn.Parameter(truncated_normal(size=(num_classes, attention_unit_size),std=0.1))
        
    def forward(self,x,omega_h):
        x_transposed = torch.permute(x,(0,2,1))
        attention_matrix = torch.matmul(
            torch.multiply(self.W_s2, omega_h),
            torch.tanh(
                torch.matmul(self.W_s1,x_transposed)
            )
            
        )
        attention_weight = F.softmax(attention_matrix, dim=1)
        attention_out = torch.matmul(attention_weight, x)
        attention_out_avg = torch.mean(attention_out, dim=1)
        attention_weight_avg = F.softmax(torch.mean(attention_weight, dim=1), dim=1)
        return attention_out_avg
    
    def __repr__(self):
        return f'TCCA Module Params: W_s1: {self.W_s1.numel()}, W_s2: {self.W_s2.numel()}'
    
class CPM(nn.Module):
    """Class Predicting Module"""
    def __init__(self,spatial_dim,num_classes,fc_hidden_size):
        super(CPM, self).__init__()
        self.W_t = nn.Parameter(truncated_normal(size=(fc_hidden_size, 2*spatial_dim),std=0.1))
        self.b_t = nn.Parameter(torch.ones(fc_hidden_size)*0.1)
        self.W_l = nn.Parameter(truncated_normal(size=(num_classes, fc_hidden_size),std=0.1))
        self.b_l = nn.Parameter(torch.ones(num_classes)*0.1)
    def forward(self,x):
        fc = F.linear(x,self.W_t,self.b_t)
        local_fc_out = F.relu(fc)
        local_logits = F.linear(local_fc_out,self.W_l,self.b_l)
        local_scores = F.sigmoid(local_logits)
        return local_scores, local_fc_out
    
    def __repr__(self):
        return f'CPM Module Params: W_t: {self.W_t.numel()}, W_l: {self.W_l.numel()}, b_t: {self.b_t.numel()}, b_l: {self.b_l.numel()}'
    
class CDM(nn.Module):
    """Class Dependency Module"""
    def __init__(self,num_classes,next_num_classes,attention_unit_size):
        super(CDM, self).__init__()
        self.G_h_implicit = None
        if next_num_classes is not None:
            self.G_h_implicit = nn.Parameter(truncated_normal(size=(num_classes, next_num_classes),std=0.1))
        
        self.next_num_classes = next_num_classes
        self.attention_unit_size = attention_unit_size
        
    def forward(self,x):
        # if last CDM Module, omega_h is not needed any more.
        if self.next_num_classes is None:
            return torch.zeros(1)
        Q_h = torch.matmul(x, self.G_h_implicit)
        omega_h = Q_h.unsqueeze(-1).repeat(1,1,self.attention_unit_size)
        #omega_h = Q_h_repeated.view(self.next_num_classes,self.attention_unit_size)
        return omega_h
    
    def __repr__(self):
        if self.G_h_implicit is None:
            return ''
        return f'CDM Module Params: G_h_implicit: {self.G_h_implicit.numel()}'
    
class HAM(nn.Module):
    """HAM Unit"""
    def __init__(self,feature_dim,num_classes,next_num_classes,attention_unit_size,fc_hidden_size):
        super(HAM, self).__init__()
        self.tcca = TCCA(feature_dim=feature_dim,num_classes=num_classes,attention_unit_size=attention_unit_size)
        self.cpm = CPM(num_classes=num_classes, fc_hidden_size=fc_hidden_size,spatial_dim=feature_dim[1])
        self.cdm = CDM(num_classes=num_classes,next_num_classes=next_num_classes,attention_unit_size=attention_unit_size)
        
        
    def forward(self,x,omega_h):
        attention_out_avg = self.tcca(x,omega_h)
        x_mean = torch.mean(x,axis=1)
        local_input = torch.cat((x_mean,attention_out_avg),dim=1)
        local_score, local_fc_out = self.cpm(local_input)
        omega_h_next = self.cdm(local_score)
        return local_score,local_fc_out, omega_h_next
    
    def __repr__(self):
        return (f'HAM Module: '
                f'\nTCCA Module: {self.tcca.__repr__()}'
                f'\nCPM Module: {self.cpm.__repr__()}'
                f'\nCDM Module: {self.cdm.__repr__()}')
    
class HybridPredictingModule(nn.Module):
    def __init__(self,fc_hidden_size,total_classes,dropout_keep_prob,alpha):
        super(HybridPredictingModule,self).__init__()
        self.W_g = nn.Parameter(truncated_normal(size=(fc_hidden_size, fc_hidden_size),std=0.1))
        self.b_g = nn.Parameter(torch.ones(fc_hidden_size)*0.1)
        self.W_m = nn.Parameter(truncated_normal(size=(total_classes,fc_hidden_size),std=0.1))
        self.b_m = nn.Parameter(torch.ones(total_classes)*0.1)
        self.drop = nn.Dropout(dropout_keep_prob)
        self.alpha = alpha
        
        
    def forward(self,local_logits,local_scores):
        #ham_out = torch.cat([local_logits.unsqueeze(1) for local_logits in local_logits_list], dim=1)
        avg_ham_out = torch.mean(local_logits,dim=1)
        avg_ham_out_fc = F.linear(avg_ham_out,self.W_g,self.b_g)
        fc_out = F.relu(avg_ham_out_fc)
        fc_out_drop = self.drop(fc_out)
        global_logits = F.linear(fc_out_drop,self.W_m,self.b_m)
        global_scores = F.sigmoid(global_logits)
        #local_scores_list = torch.cat(local_scores_list,dim=1)
        scores = torch.add(self.alpha*global_scores,(1. - self.alpha)*local_scores)
        return scores, global_logits
    
    def __repr__(self):
        return (f'HybridPredicting Module Params: W_g: {self.W_g.numel()}, W_m: {self.W_m.numel()}, b_g: {self.b_g.numel()}, b_m: {self.b_m.numel()}')
        

class HybridPredictingModuleHighway(nn.Module):
    def __init__(self,num_layers,num_highway_layers,fc_hidden_size,total_classes,dropout_keep_prob,alpha):
        super(HybridPredictingModuleHighway,self).__init__()
        self.highway = HighwayLayer(input_size=fc_hidden_size,num_layers=num_highway_layers)
        self.W_highway = nn.Parameter(truncated_normal(size=(fc_hidden_size, fc_hidden_size*num_layers),std=0.1))
        self.b_highway = nn.Parameter(torch.ones(fc_hidden_size)*0.1)
        self.W_global_pred = nn.Parameter(truncated_normal(size=(total_classes,fc_hidden_size),std=0.1))
        self.b_global_pred = nn.Parameter(torch.ones(total_classes)*0.1)
        self.highway_drop = nn.Dropout(dropout_keep_prob)
        self.alpha = alpha
        
    def forward(self,local_logits_list,local_scores):
        ham_out = torch.cat(local_logits_list,dim=1)
        fc = F.linear(ham_out,self.W_highway,self.b_highway)
        fc_out = F.relu(fc)
        highway = self.highway(fc_out)
        highway_drop_out = self.highway_drop(highway)
        global_logits = F.linear(highway_drop_out,self.W_global_pred,self.b_global_pred)
        global_scores = F.sigmoid(global_logits)
        #local_scores_list = torch.cat(local_scores_list,dim=1)
        scores = torch.add(self.alpha*global_scores,(1. - self.alpha)*local_scores)
        return scores, global_logits
    
    def __repr__(self):
        return (f'HybridPredicting Module Params: W_highway: {self.W_highway.numel()}, W_global_pred: {self.W_global_pred.numel()}, b_highway: {self.b_highway.numel()}, b_global_pred: {self.b_global_pred.numel()}'
                f'HighwayLayer Module Params: {self.highway.__repr__()}')
        
class HighwayLayer(nn.Module):
    def __init__(self, input_size, num_layers=1, bias=-2.0):
        super(HighwayLayer, self).__init__()
        self.num_layers = num_layers

        # Define linear layers for transform gate (t) and carry gate (1 - t)
        self.transform_gate = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(num_layers)])
        self.carry_gate = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(num_layers)])
        
        # Initialize the bias for the carry gate
        for layer in self.carry_gate:
            layer.bias.data.fill_(bias)

    def forward(self, input_):
        output = input_

        for idx in range(self.num_layers):
            # Transform gate (t)
            transform_gate = torch.sigmoid(self.transform_gate[idx](input_))
            
            # Carry gate (1 - t)
            carry_gate = 1. - transform_gate
            
            # Linear transformation with ReLU activation
            transformed_output = F.relu(self.carry_gate[idx](input_))
            
            # Highway network formula: z = t * h + (1 - t) * x
            output = transform_gate * transformed_output + carry_gate * output
            input_ = output

        return output
    
    def __repr__(self):
        transform_params = sum(p.numel() for p in self.transform_gate.parameters())
        carry_params = sum(p.numel() for p in self.carry_gate.parameters())
        return f'HighwayLayer(input_size={self.transform_gate[0].weight.size(1)}, num_layers={self.num_layers}, transform_params={transform_params}, carry_params={carry_params})'
        
class HmcNet(nn.Module):
    """A HARNN for image classification."""
    def __init__(self,feature_dim,attention_unit_size,fc_hidden_size,highway_num_layers, num_classes_list, total_classes, freeze_backbone,l2_reg_lambda=0.0,dropout_keep_prob=0.5,alpha=0.5,beta=0.5,device=None):
        super(HmcNet,self).__init__()
        resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        self.backbone = torch.nn.Sequential(*(list(resnet50.children())[:len(list(resnet50.children()))-1]))
        self.ham_modules = nn.ModuleList()
        self.feature_dim = feature_dim
        self.attention_unit_size = attention_unit_size
        self.fc_hidden_size = fc_hidden_size
        self.num_classes_list = num_classes_list
        self.total_classes = total_classes
        self.l2_reg_lambda = l2_reg_lambda
        self.alpha = alpha
        self.device = device
        self.beta = beta
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        for i in range(len(num_classes_list)):
            if i == len(num_classes_list)-1:
                #If its the last HAM Module, the last omega_h of CDM ist not needed.
                self.ham_modules.append(HAM(feature_dim=feature_dim,num_classes=num_classes_list[i],next_num_classes=None,attention_unit_size=attention_unit_size,fc_hidden_size=fc_hidden_size))
                break
            self.ham_modules.append(HAM(feature_dim=feature_dim,num_classes=num_classes_list[i],next_num_classes=num_classes_list[i+1],attention_unit_size=attention_unit_size,fc_hidden_size=fc_hidden_size))
        
        self.hybrid_predicting_module = HybridPredictingModuleHighway(fc_hidden_size=fc_hidden_size,num_layers=len(num_classes_list),num_highway_layers=highway_num_layers,total_classes=total_classes,dropout_keep_prob=dropout_keep_prob,alpha=alpha)
        
    def forward(self,x):
        feature_extractor_out = self.backbone(x)
        num_channels,spatial_dim1, spatial_dim2 = feature_extractor_out.shape[1:]
        feature_extractor_out = feature_extractor_out.view(-1, num_channels, spatial_dim1 * spatial_dim2)
        omega_h = torch.ones(self.num_classes_list[0],self.attention_unit_size).to(self.device)
        local_score_list = []
        local_logit_list = []
        for ham_module in self.ham_modules:
            local_score, local_logit, omega_h = ham_module(feature_extractor_out,omega_h)
            local_score_list.append(local_score)
            local_logit_list.append(local_logit)
        
        #local_logits= torch.cat([local_logits.unsqueeze(1) for local_logits in local_logit_list], dim=1)
        local_scores = torch.cat(local_score_list,dim=1)
        
        
        scores, global_logits = self.hybrid_predicting_module(local_logit_list,local_scores)
        return scores, local_score_list, global_logits
    
    def __repr__(self):
        str = f'{self.hybrid_predicting_module.__repr__()}'
        for i in range(len(self.ham_modules)):
            str += f'{self.ham_modules[i].__repr__()}\n'
        return str
    
# Define Loss for HmcNet.
class HmcNetLoss(nn.Module):
    def __init__(self,beta,l2_lambda,model,explicit_hierarchy,device=None):
        super(HmcNetLoss, self).__init__()
        self.beta = beta
        self.l2_lambda = l2_lambda
        self.model = model
        self.explicit_hierarchy = explicit_hierarchy
        self.device = device
        
    
    def forward(self,predictions,targets):
        """Calculation of the HmcNet loss."""

        def _local_loss(local_scores_list,local_target_list):
            """Calculation of the Local loss."""
            losses = torch.zeros(len(local_scores_list)).to(self.device)
            for i in range(len(local_scores_list)):
                local_scores = local_scores_list[i]
                local_targets = local_target_list[i]
                loss = F.binary_cross_entropy(local_scores, local_targets)
                mean_loss = torch.mean(loss)
                losses[i] = mean_loss
            return torch.sum(losses)

        def _global_loss(global_logits,global_target):
            """Calculation of the Global loss."""
            global_scores = torch.sigmoid(global_logits)
            loss = F.binary_cross_entropy(global_scores,global_target)
            mean_loss = torch.mean(loss)
            return mean_loss

        def _hierarchy_constraint_loss(global_logits):
            """Calculate the Hierarchy Constraint loss."""
            global_scores = torch.sigmoid(global_logits)
            hierarchy_losses = torch.zeros(global_scores.shape[0]).to(self.device)
            for i, global_score in enumerate(global_scores):
                mask = self.explicit_hierarchy == 1
                score_diff = global_score.unsqueeze(0) - global_score.unsqueeze(1)                
                loss = self.beta * torch.max(torch.tensor(0.0).to(self.device), score_diff[mask]) ** 2
                temp_loss = torch.sum(loss)
                hierarchy_losses[i] = temp_loss
            return torch.mean(hierarchy_losses)

        def _l2_loss(model,l2_reg_lambda):
            """Calculation of the L2-Regularization loss."""
            
            l2_loss = torch.tensor(0.,dtype=torch.float32).to(self.device)
            for param in model.parameters():
                if param.requires_grad == True:
                    l2_loss += torch.norm(param,p=2)**2
            return torch.tensor(l2_loss*l2_reg_lambda,dtype=torch.float32)
        local_scores_list,global_logits  = predictions[0], predictions[1]
        local_target, global_target = targets[0], targets[1]
        global_loss = _global_loss(global_logits=global_logits,global_target=global_target)
        local_loss = _local_loss(local_scores_list=local_scores_list,local_target_list=local_target)
        l2_loss = _l2_loss(model=self.model,l2_reg_lambda=self.l2_lambda)
        hierarchy_loss = _hierarchy_constraint_loss(global_logits=global_logits)
        loss = torch.sum(torch.stack([global_loss,local_loss,l2_loss,hierarchy_loss]))
        return loss
        

