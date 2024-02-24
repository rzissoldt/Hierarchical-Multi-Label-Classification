import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
def calculate_filter_pow(num_layers):
    """
    Calculate the number of filters for each layer based on the number of layers.
    The pattern is as follows
    ...

    :param num_layers: Number of layers
    :return: List containing the number of filters for each layer
    """
    filters = []
    
    for i in range(num_layers):
        temp_filter = []
        if i == 0:
            temp_filter.append(128)
        elif i == 1:
            temp_filter.append(128)
            temp_filter.append(256)
        else:
            temp_filter.append(128)
            temp_filter.append(256)
            temp_filter.append(512)
        filters.append(temp_filter)
        
    return filters

def squash(s, axis=-1):
    """
    Non-linear squashing function to manipulate the length of the capsule vectors.
    
    :param s: input tensor containing capsule vectors
    :param axis: If `axis` is a Python integer, the input is considered a batch of vectors,
                 and `axis` determines the axis in `tensor` over which to compute squash.
    :return: a Tensor with the same shape as input vectors
    """
    squared_norm = torch.sum(s ** 2, dim=axis, keepdim=True)
    safe_norm = torch.sqrt(squared_norm + torch.finfo(squared_norm.dtype).eps)
    squash_factor = squared_norm / (1. + squared_norm)
    unit_vector = s / safe_norm
    return squash_factor * unit_vector


def safe_norm(s, axis=-1, keepdim=False):
    """
    Safe computation of vector 2-norm
    
    :param s: input tensor
    :param axis: If `axis` is a Python integer, the input is considered a batch 
                 of vectors, and `axis` determines the axis in `tensor` over which to 
                 compute vector norms.
    :param keepdim: If True, the axis indicated in `axis` are kept with size 1.
                    Otherwise, the dimensions in `axis` are removed from the output shape.
    :return: A `Tensor` of the same type as tensor, containing the vector norms. 
             If `keepdim` is True then the rank of output is equal to
             the rank of `tensor`. If `axis` is an integer, the rank of `output` is 
             one less than the rank of `tensor`.
    """
    squared_norm = torch.sum(s ** 2, dim=axis, keepdim=keepdim)
    return torch.sqrt(squared_norm + torch.finfo(squared_norm.dtype).eps)

class Decoder(nn.Module):
    def __init__(self, input_dim,target_shape,fc_hidden_size,num_layers,output_dim):
        super(Decoder, self).__init__()
        hidden_layers = []
        for i in range(num_layers):
            if i == 0:
                hidden_layers.append(nn.Linear(in_features=input_dim,out_features=fc_hidden_size))
            else:
                hidden_layers.append(nn.Linear(in_features=fc_hidden_size,out_features=fc_hidden_size))
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.output_layer = nn.Linear(in_features=fc_hidden_size,out_features=output_dim)
        self.target_shape = target_shape
        
    def forward(self,x):
        for i in range(len(self.hidden_layers)):
            x = self.hidden_layers[i](x)
            x = F.relu(x)
        x = F.sigmoid(self.output_layer(x))
        return x.view(*self.target_shape)
        
class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=64,kernel_size=(3,3),stride=(1,1),padding=1)
        self.batchnorm1 = nn.BatchNorm2d(num_features=64)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),stride=(1,1),padding=1)
        self.batchnorm2 = nn.BatchNorm2d(num_features=64)
        
    
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.batchnorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchnorm2(x)
        return x

class PrimaryCapsule(nn.Module):
    def __init__(self,in_channels,pcap_n_dims,num_classes_list):
        super(PrimaryCapsule, self).__init__()
        def conv_block(in_channels,filter_count_list):
            module_list = []
            for i in range(len(filter_count_list)):
                if i == 0:
                    module_list.append(nn.Conv2d(in_channels=in_channels,out_channels=filter_count_list[i],kernel_size=(3,3),padding=1))
                else:
                    module_list.append(nn.Conv2d(in_channels=filter_count_list[i-1],out_channels=filter_count_list[i],kernel_size=(3,3),padding=1))
                module_list.append(nn.ReLU())
                module_list.append(nn.BatchNorm2d(num_features=filter_count_list[i]))
                module_list.append(nn.Conv2d(in_channels=filter_count_list[i],out_channels=filter_count_list[i],kernel_size=(3,3),padding=1))
                module_list.append(nn.ReLU())
                module_list.append(nn.BatchNorm2d(num_features=filter_count_list[i]))
                module_list.append(nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)))
            
            block = nn.Sequential(*module_list)
            return block
        filter_list = calculate_filter_pow(len(num_classes_list))
        conv_block_list = []
        for i in range(len(filter_list)):
            conv_block_list.append(conv_block(in_channels=in_channels,filter_count_list=filter_list[i]))
        self.conv_blocks = nn.ModuleList(conv_block_list)
        self.pcap_n_dims = pcap_n_dims
    def forward(self,x):
        outputs = []
        for conv_block in self.conv_blocks:
            pri_capsule_out = conv_block(x)
            outputs.append(pri_capsule_out)
        
        squashed_outputs = []
        for output in outputs:
           
            total_elements = np.prod(output.shape[1:])
            # Calculate the size of the second dimension
            second_dim_size = total_elements // 8
            # Reshape the tensor
            reshaped_output = output.view(-1,8, second_dim_size)  # -1 lets PyTorch calculate the size automatically

            squash_output = squash(reshaped_output)

            squashed_outputs.append(squash_output)
            
        return squashed_outputs           
class SecondaryCapsule(nn.Module):
    """
    The Secondary Capsule layer With Dynamic Routing Algorithm. 
    input shape = [None, input_num_capsule, input_dim_capsule] 
    output shape = [None, num_capsule, dim_capsule]
    :param n_caps: number of capsules in this layer
    :param n_dims: dimension of the output vectors of the capsules in this layer
    """
    def __init__(self, in_channels,pcap_n_dims, n_caps, n_dims, routings=2):
        super(SecondaryCapsule, self).__init__()
        self.n_caps = n_caps
        self.n_dims = n_dims
        self.routings = routings
        self.in_channels = in_channels
        self.pcap_n_dims = pcap_n_dims
        # Initialize transformation matrix
        self.W = nn.Parameter(torch.randn(1, in_channels, self.n_caps, self.n_dims, pcap_n_dims))

    def forward(self, x):
        # Predict output vector
        batch_size = x.size(0)
        caps1_n_caps = x.size(1)
        W_tiled = self.W.repeat(batch_size, 1, 1, 1, 1)
        caps1_output_expanded = x.unsqueeze(-1)
        caps1_output_tile = caps1_output_expanded.unsqueeze(2)
        caps1_output_tiled = caps1_output_tile.repeat(1, 1, self.n_caps, 1, 1)
        caps2_predicted = torch.matmul(W_tiled, caps1_output_tiled)

        # Routing by agreement
        # Initialize routing weights
        raw_weights = torch.zeros(batch_size, caps1_n_caps, self.n_caps, 1, 1, dtype=torch.float32)
        for i in range(self.routings):
            routing_weights = F.softmax(raw_weights, dim=2)
            weighted_predictions = routing_weights * caps2_predicted
            weighted_sum = weighted_predictions.sum(dim=1, keepdim=True)
            caps2_output_round_1 = squash(weighted_sum, axis=-2)
            caps2_output_squeezed = caps2_output_round_1.squeeze(dim=[1, 4])
            if i < self.routings - 1:
                caps2_output_round_1_tiled = caps2_output_round_1.repeat(1, caps1_n_caps, 1, 1, 1)
                agreement = torch.matmul(caps2_predicted, caps2_output_round_1_tiled.transpose(1, 0))
                raw_weights_round_2 = raw_weights + agreement
                raw_weights = raw_weights_round_2
        return caps2_output_squeezed

class LengthLayer(nn.Module):
    """
    Compute the length of capsule vectors.
    inputs: shape=[None, num_capsule, dim_vector]
    output: shape=[None, num_capsule]
    """
    def __init__(self):
        super(LengthLayer,self).__init__()
        
    def forward(self, x):
        y_proba = safe_norm(x, axis=-1)
        return y_proba

class Mask(nn.Module):
    """
    Mask a Tensor with the label during training 
    and mask a Tensor with predicted label during test/inference
    input shapes
      X shape = [None, num_capsule, dim_vector] 
      y_true shape = [None, num_capsule] 
      y_pred shape = [None, num_capsule]
    output shape = [None, num_capsule * dim_vector]
    """
    def __init__(self,input_shape, is_training=None):
        super(Mask, self).__init__()
        self.n_caps = input_shape[0][1]
        self.n_dims = input_shape[0][2]
        self.is_training = is_training
    def forward(self, input):
        x, y_true, y_proba = input
        if self.is_training:
            reconstruction_mask = y_true
        else:
            y_proba_argmax = torch.argmax(y_proba, dim=1)
            y_pred = F.one_hot(y_proba_argmax, num_classes=self.n_caps).float()
            reconstruction_mask = y_pred

        reconstruction_mask_reshaped = reconstruction_mask.unsqueeze(-1)
        caps2_output_masked = x * reconstruction_mask_reshaped
        decoder_input = caps2_output_masked.view(-1, self.n_caps * self.n_dims)
        return decoder_input
    
class MarginLoss(nn.Module):
    """
    Compute margin loss.
    y_true shape [None, n_classes] 
    y_pred shape [None, num_capsule] = [None, n_classes]
    """
    def __init__(self, m_plus=0.9, m_minus=0.1, lambda_=0.5):
        super(MarginLoss, self).__init__()
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lambda_ = lambda_

    def forward(self, y_pred, y_true):
        present_error_raw = torch.square(F.relu(self.m_plus - y_pred))
        absent_error_raw = torch.square(F.relu(y_pred - self.m_minus))
        L = torch.add(y_true * present_error_raw,self.lambda_ * (1.0 - y_true) * absent_error_raw)
        total_marginloss = torch.mean(torch.sum(L, dim=1))
        return total_marginloss
class HCapsNet(nn.Module):
    def __init__(self,feature_dim,input_shape,num_classes_list,pcap_n_dims,scap_n_dims,fc_hidden_size,num_layers):
        super(HCapsNet,self).__init__()
        
        self.encoder = Encoder(in_channels=input_shape[2])
        self.feature_dim = feature_dim
        self.pcap_n_dims = pcap_n_dims
        self.scap_n_dims = scap_n_dims
        self.primary_capsule = PrimaryCapsule(in_channels=64,pcap_n_dims=pcap_n_dims,num_classes_list=num_classes_list)
        
        secondary_capsules = []
        length_layers = []
        masks = []
        decoders = []
        n_output = np.prod(input_shape)
        secondary_capsule_input_dim = None
        for i in range(len(num_classes_list)):
            if i == 0:
                secondary_capsule_input_dim = int(128*((input_shape[0]/2)**2)/pcap_n_dims)
            elif i == 1:
                secondary_capsule_input_dim = int(256*((input_shape[0]/4)**2)/pcap_n_dims)
            else:
                secondary_capsule_input_dim = int(512*((input_shape[0]/8)**2)/pcap_n_dims)
            
            secondary_capsules.append(SecondaryCapsule(in_channels=secondary_capsule_input_dim,pcap_n_dims=pcap_n_dims,n_caps=num_classes_list[i],n_dims=scap_n_dims))
            length_layers.append(LengthLayer())
            masks.append(Mask(input_shape=feature_dim))
            decoders.append(Decoder(input_shape=self.scap_n_dims*num_classes_list[i],fc_hidden_size=fc_hidden_size, num_layers=num_layers,output_dim=n_output))
        self.secondary_capsules_modules = nn.ModuleList(secondary_capsules)
        #self.length_layers = nn.ModuleList(length_layers)
        #self.masks = nn.ModuleList(masks)
        #self.decoders = nn.ModuleList(decoders)
        #self.concatenated = nn.Sequential(
        #    nn.Linear(3 * n_output, 4),
        #    nn.ReLU(),
        #    nn.Linear(4, 3)
        #)
    def forward(self,x):
        image, y_true = x
        
        feature_out = self.encoder(image)
        squashed_outputs = self.primary_capsule(feature_out)
        secondary_capsule_outputs = []
        length_layer_outputs = []
        decoder_inputs = []
        decoder_outputs = []
        for i in range(len(self.secondary_capsules_modules)):
            secondary_capsule_output = self.secondary_capsules_modules[i](squashed_outputs[i])
            length_layer_output = self.length_layers[i](secondary_capsule_output)
            secondary_capsule_outputs.append(secondary_capsule_output)
            length_layer_outputs.append(length_layer_output)
            x = secondary_capsule_output, y_true[i], length_layer_output
            decoder_input = self.mask_layers[i](x)
            decoder_inputs.append(decoder_input)
            decoder_output = self.decoders[i](decoder_input)
            decoder_outputs.append(decoder_output)
        cat_decoder_output = torch.cat(decoder_outputs)
        final_output = self.concatenated(cat_decoder_output)
        return length_layer_output, final_output



