import torch
import torch.nn as nn
from model.backbone import Backbone
def get_constr_out(x, R):
    """ Given the output of the neural network x returns the output of MCM given the hierarchy constraint expressed in the matrix R """
    c_out = x.double()
    c_out = c_out.unsqueeze(1)
    c_out = c_out.expand(len(x),R.shape[1], R.shape[1]) # H-strich
    R_batch = R.expand(len(x),R.shape[1], R.shape[1]) # Mask M
    final_out, _ = torch.max(R_batch*c_out.double(), dim = 2)
    return final_out



class ConstrainedFFNNModel(nn.Module):
    """ C-HMCNN(h) model - during training it returns the not-constrained output that is then passed to MCLoss """
    def __init__(self, output_dim,R, args):
        super(ConstrainedFFNNModel, self).__init__()
        self.backbone = Backbone(global_average_pooling_active=True)
        self.nb_layers = args.num_layers
        self.feature_dim = args.feature_dim_backbone[0]
        self.hidden_dim = args.fc_dim
        self.R = R
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
        if self.training:
            constrained_out = x
        else:
            constrained_out = get_constr_out(x, self.R)
        return constrained_out
        
    
class ConstrainedFFNNModelLoss(nn.Module):
    def __init__(self,l2_lambda,device):
        super(ConstrainedFFNNModelLoss, self).__init__()
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
        global_loss = self.criterion(predictions,targets)
        l2_loss = _l2_loss(model=model,l2_reg_lambda=self.l2_lambda)
        loss = torch.sum(torch.stack([global_loss,l2_loss]))
        return loss, global_loss, l2_loss   