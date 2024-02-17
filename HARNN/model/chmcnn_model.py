import torch
import torch.nn as nn
def get_constr_out(x, R):
    """ Given the output of the neural network x returns the output of MCM given the hierarchy constraint expressed in the matrix R """
    c_out = x.double()
    c_out = c_out.unsqueeze(1)
    c_out = c_out.expand(len(x),R.shape[1], R.shape[1])
    R_batch = R.expand(len(x),R.shape[1], R.shape[1])
    final_out, _ = torch.max(R_batch*c_out.double(), dim = 2)
    return final_out



class ConstrainedFFNNModel(nn.Module):
    """ C-HMCNN(h) model - during training it returns the not-constrained output that is then passed to MCLoss """
    def __init__(self, output_dim,R, args):
        super(ConstrainedFFNNModel, self).__init__()
        resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        self.backbone = nn.Sequential(*(list(resnet50.children())[:len(list(resnet50.children()))-1]))
        self.nb_layers = args.num_layers
        self.feature_dim = args.feature_dim_backbone[0]
        self.hidden_dim = args.fc_dim
        self.R = R
        if args.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        fc = []
        for i in range(self.nb_layers):
            if i == 0:
                fc.append(nn.Linear(self.feature_dim, self.hidden_dim))
            elif i == self.nb_layers-1:
                fc.append(nn.Linear(self.hidden_dim, output_dim))
            else:
                fc.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.fc = nn.ModuleList(fc)
        
        self.drop = nn.Dropout(args.dropout_rate)
        
        
        self.sigmoid = nn.Sigmoid()
        if args.activation_func == 'tanh':
            self.f = nn.Tanh()
        else:
            self.f = nn.ReLU()
        
    def forward(self, x):
        feature_extractor_out = self.backbone(x)
        num_channels,spatial_dim1, spatial_dim2 = feature_extractor_out.shape[1:]
        feature_extractor_out = feature_extractor_out.view(-1, num_channels, spatial_dim1 * spatial_dim2)
        x = torch.squeeze(feature_extractor_out,dim=2)
        for i in range(self.nb_layers):
            if i == self.nb_layers-1:
                x = self.sigmoid(self.fc[i](x))
            else:
                x = self.f(self.fc[i](x))
                x = self.drop(x)
        
        return x