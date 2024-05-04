import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.hcapsnet_model import squash, safe_norm, SecondaryCapsule, LengthLayer, MarginLoss
from model.backbone import Backbone
from scipy.stats import chi2

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

class PrimaryCapsule(nn.Module):
    def __init__(self,pcap_n_dims):
        super(PrimaryCapsule, self).__init__()
        self.pcap_n_dims = pcap_n_dims
    def forward(self,x):
        
           
        total_elements = np.prod(x.shape[1:])
        # Calculate the size of the second dimension
        second_dim_size = total_elements // self.pcap_n_dims
        # Reshape the tensor
        reshaped_output = x.view(-1,self.pcap_n_dims, second_dim_size)  # -1 lets PyTorch calculate the size automatically

        squashed_output = squash(reshaped_output).permute(0,2,1)

            
        return squashed_output


class BUHCapsNet(nn.Module):
    def __init__(self,pcap_n_dims, scap_n_dims, num_classes_list,routings,args,device=None):
        super(BUHCapsNet, self).__init__()
        self.backbone = Backbone(global_average_pooling_active=True)
        # Print the structure
        #print(self.backbone.parameters[:2])
        for name, module in self.backbone.named_children():
            print(name)
            print(module)
        if args.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        self.primary_capsule = PrimaryCapsule(pcap_n_dims)  # Assuming 8 primary capsules
        secondary_capsules_list = []
        secondary_capsules_list.append(SecondaryCapsule(in_channels=256,pcap_n_dims=pcap_n_dims,n_caps=num_classes_list[-1],routings=routings,n_dims=scap_n_dims,device=device))
        
        secondary_capsules_list.extend([SecondaryCapsule(in_channels=num_classes_list[i+1],pcap_n_dims=scap_n_dims,n_caps=num_classes_list[i],routings=routings,n_dims=scap_n_dims,device=device) for i in range(len(num_classes_list)-2,-1,-1)])
        #secondary_capsules_list.append(SecondaryCapsule(num_routes=12544,in_channels=pcap_n_dims,num_capsules=num_classes_list[-1],routings=routings,out_channels=scap_n_dims,device=device))
        #secondary_capsules_list.extend([SecondaryCapsule(num_routes=num_classes_list[i+1],in_channels=scap_n_dims,num_capsules=num_classes_list[i],routings=routings,out_channels=scap_n_dims,device=device) for i in range(len(num_classes_list)-2,-1,-1)])
        self.secondary_capsules = nn.ModuleList(secondary_capsules_list)
        #print(count_parameters(secondary_capsule) for secondary_capsule in self.secondary_capsules)
        self.length_layer = LengthLayer()
        
    def forward(self,x):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        feature_output = self.backbone(x)
        end.record()
        torch.cuda.synchronize()
        print('To Backbone Forward:',start.elapsed_time(end))
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        primary_capsule_output = self.primary_capsule(feature_output)
        end.record()
        torch.cuda.synchronize()
        print('To Primary Capsule:',start.elapsed_time(end))
        output_list = []
        for i in range(len(self.secondary_capsules)):
            if i == 0:
                secondary_capsule_out = self.secondary_capsules[i](primary_capsule_output)
            else:
                secondary_capsule_out = self.secondary_capsules[i](secondary_capsule_out)
            output_list.append(self.length_layer(secondary_capsule_out))
                
        return output_list[::-1]
    
class BUHCapsNetLoss(nn.Module):
    def __init__(self,num_classes_list, m_plus=0.9, m_minus=0.1, lambda_=0.5,device=None):
        super(BUHCapsNetLoss, self).__init__()
        self.margin_loss = MarginLoss(m_plus = m_plus,m_minus = m_minus,lambda_ = lambda_)
        self.device = device
        self.initial_loss_weights = [1./len(num_classes_list) for i in range(len(num_classes_list))]
        self.current_loss_weights = [1./len(num_classes_list) for i in range(len(num_classes_list))]
    def forward(self,x):
        y_pred,y_true= x
        margin_losses = torch.zeros(len(y_pred)).to(self.device)
        for i in range(len(y_pred)):
            x_margin_loss = y_pred[i],y_true[i]
            margin_loss = self.margin_loss(x_margin_loss)
            margin_losses[i] = margin_loss*self.current_loss_weights[i]
        
        margin_loss_sum = torch.sum(margin_losses)
        
        return margin_loss_sum
    def update_loss_weights(self,layer_accuracies):
        taus = [(1-layer_accuracies[i]) * self.initial_loss_weights[i] for i in range(len(self.current_loss_weights))]
        for i in range(len(taus)):
            self.current_loss_weights[i] = taus[i]/sum(taus)

        
class LambdaUpdater():
    def __init__(self, num_layers, initial_k, final_k, num_epochs):
        self.num_layers = num_layers
        self.initial_k = initial_k
        self.final_k = final_k
        self.current_epoch = 0
        self.k = initial_k
        self.num_epochs = num_epochs
        self.step_size = (self.final_k - self.initial_k) / num_epochs

    def update_lambdas(self):
        if self.current_epoch < self.num_epochs:
            self.k = self.initial_k + self.step_size * self.current_epoch

    def get_lambda_values(self):
        x_values = np.linspace(10**-16, self.final_k, self.num_layers)  # Generate x values
        density_values = chi2.pdf(x_values, self.k)  # Evaluate density function at x values
        lambda_values = density_values / np.sum(density_values)  # Normalize density values
        return lambda_values

    def next_epoch(self):
        self.current_epoch += 1

