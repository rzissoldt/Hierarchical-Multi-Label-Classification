import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.hcapsnet_model import squash, safe_norm, SecondaryCapsule, LengthLayer, MarginLoss
from model.backbone import Backbone
from scipy.stats import chi2
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

class MyResNet50(ResNet):
    def __init__(self):
        super(MyResNet50, self).__init__(Bottleneck, [3, 4, 6, 3])
        self.load_state_dict(resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).state_dict())
    def forward(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        return x
    
#class PrimaryCapsule(nn.Module):
#    def __init__(self,pcap_n_dims):
#        super(PrimaryCapsule, self).__init__()
#        self.pcap_n_dims = pcap_n_dims
#    def forward(self,x):
#        
#           
#        total_elements = np.prod(x.shape[1:])
#        # Calculate the size of the second dimension
#        second_dim_size = total_elements // self.pcap_n_dims
#        # Reshape the tensor
#        reshaped_output = x.view(-1,self.pcap_n_dims, second_dim_size)  # -1 lets PyTorch calculate the size automatically
#
#        squashed_output = squash(reshaped_output).permute(0,2,1)
#
#            
#        return squashed_output
class PrimaryCapsule(nn.Module):
    def __init__(self, num_capsules=8, in_channels=512, out_channels=128, kernel_size=3, num_routes=32 * 6 * 6):
        super(PrimaryCapsule, self).__init__()
        self.num_routes = num_routes
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=0)
            for _ in range(num_capsules)])

    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), self.num_routes, -1)
        return squash(u)




class SecondaryCapsule(nn.Module):
    def __init__(self, num_capsules=10, num_routes=32 * 6 * 6, in_channels=8, out_channels=16,routings=3,device=None):
        super(SecondaryCapsule, self).__init__()

        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules
        self.routings = routings
        self.device = device
        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.squeeze()
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)

        W = torch.cat([self.W] * batch_size, dim=0)
        
        u_hat = torch.matmul(W, x)

        b_ij = torch.autograd.Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1)).to(self.device)

        
        for iteration in range(self.routings):
            c_ij = F.softmax(b_ij, dim=1)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = squash(s_j)

            if iteration < self.routings - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

        return v_j.squeeze(1)

    

class BUHCapsNet(nn.Module):
    def __init__(self,pcap_n_dims, scap_n_dims, num_classes_list,routings,args,device=None):
        super(BUHCapsNet, self).__init__()
        self.resnet18 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*(list(self.resnet18.children())[:len(list(self.resnet18.children()))-2]))
        if args.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        
        self.primary_capsule = PrimaryCapsule()
        secondary_capsules_list = []
        secondary_capsules_list.append(SecondaryCapsule(in_channels=pcap_n_dims,num_capsules=num_classes_list[-1],routings=routings,out_channels=scap_n_dims,device=device))
        secondary_capsules_list.extend([SecondaryCapsule(num_routes=num_classes_list[i+1],in_channels=scap_n_dims,num_capsules=num_classes_list[i],routings=routings,out_channels=scap_n_dims,device=device) for i in range(len(num_classes_list)-2,-1,-1)])
        self.secondary_capsules = nn.ModuleList(secondary_capsules_list)
        self.length_layer = LengthLayer()
        
    def forward(self,x):
        feature_output = self.backbone(x)
        primary_capsule_output = self.primary_capsule(feature_output)
        output_list = []
        for i in range(len(self.secondary_capsules)):
            if i == 0:
                secondary_capsule_out = self.secondary_capsules[i](primary_capsule_output)
            else:
                secondary_capsule_out = self.secondary_capsules[i](secondary_capsule_out)
            
            output_list.append(self.length_layer(secondary_capsule_out.squeeze()))
                
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

        

