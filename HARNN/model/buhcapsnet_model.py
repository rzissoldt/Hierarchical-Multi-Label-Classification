import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.hcapsnet_model import squash, safe_norm, SecondaryCapsule, LengthLayer, MarginLoss
from model.backbone import Backbone
from scipy.stats import chi2
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        
        # Sub-block 1
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Sub-block 2
        self.conv2_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Sub-block 3
        self.conv3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Sub-block 4
        self.conv4_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Sub-block 5
        self.conv5_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Activation function
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Sub-block 1
        x = self.relu(self.bn1_1(self.conv1_1(x)))
        x = self.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        
        # Sub-block 2
        x = self.relu(self.bn2_1(self.conv2_1(x)))
        x = self.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        
        # Sub-block 3
        x = self.relu(self.bn3_1(self.conv3_1(x)))
        x = self.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)
        
        # Sub-block 4
        x = self.relu(self.bn4_1(self.conv4_1(x)))
        x = self.relu(self.bn4_2(self.conv4_2(x)))
        x = self.pool4(x)
        
        # Sub-block 5
        x = self.relu(self.bn5_1(self.conv5_1(x)))
        x = self.relu(self.bn5_2(self.conv5_2(x)))
        x = self.pool5(x)
        
        return x
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
        self.backbone = Backbone(global_average_pooling_active=False)
        if args.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        self.primary_capsule = PrimaryCapsule(pcap_n_dims)  # Assuming 8 primary capsules
        secondary_capsules_list = []
        secondary_capsules_list.append(SecondaryCapsule(in_channels=12544,pcap_n_dims=pcap_n_dims,n_caps=num_classes_list[-1],routings=routings,n_dims=scap_n_dims,device=device))
        secondary_capsules_list.extend([SecondaryCapsule(in_channels=num_classes_list[i+1],pcap_n_dims=scap_n_dims,n_caps=num_classes_list[i],routings=routings,n_dims=scap_n_dims,device=device) for i in range(len(num_classes_list)-2,-1,-1)])
        print(secondary_capsules_list)
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

