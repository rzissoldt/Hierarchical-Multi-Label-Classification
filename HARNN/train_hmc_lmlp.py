
# -*- coding:utf-8 -*-
__author__ = 'Ruben'
import numpy as np
import os,json,random
import sys
import time
import logging
import torch, copy
import math
from torchsummary import summary
import torch.optim as optim
# PyTorch TensorBoard support

from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from torchvision import transforms



# Add the parent directory to the Python path
sys.path.append('../')
from utils import xtree_utils as xtree
from utils import data_helpers as dh
from utils import param_parser as parser
from HARNN.model.hmc_lmlp_model import HmcLMLP, HmcLMLPLoss
from HARNN.dataset.hmc_lmlp_dataset import HmcLMLPDataset
from HARNN.trainer.hmc_lmlp_trainer import HmcLMLPTrainer
import torch.nn as nn
from utils.xtree_utils import generate_dicts_per_level
import warnings

# Ignore specific warning types
warnings.filterwarnings("ignore", category=UserWarning)




def train_hmc_lmlp(args):        
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available!")

        # Check if PyTorch is using CUDA
        if torch.cuda.current_device() != -1:
            print("PyTorch is using CUDA!")
        else:
            print("PyTorch is not using CUDA.")
    else:
        print("CUDA is not available!")
    
    # Checks if GPU Support ist active
    device = torch.device("cuda") if args.gpu else torch.device("cpu")
   
    # Load Input Data
    hierarchy = xtree.load_xtree_json(args.hierarchy_file)
    hierarchy_dicts = xtree.generate_dicts_per_level(hierarchy)
    num_classes_list = dh.get_num_classes_from_hierarchy(hierarchy_dicts)
    explicit_hierarchy = dh.generate_hierarchy_matrix_from_tree(hierarchy)
    image_dir = args.image_dir
    

    # Define Model 
    model = HmcLMLP(feature_dim=args.feature_dim_backbone,backbone_fc_hidden_size=args.backbone_dim,fc_hidden_size=args.fc_dim,num_classes_list=num_classes_list,freeze_backbone=args.freeze_backbone,device=device).to(device)
    model_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model Parameter Count:{model_param_count}')
    print(f'Total Classes: {sum(num_classes_list)}')
    print(f'Num Classes List: {num_classes_list}')
    
    # Define Optimzer and Scheduler
    if args.optimizer == 'adam':    
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    else:
        print(f'{args.optimizer} is not a valid optimizer. Quit Program.')
        return
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_rate)
    model.eval().to(device)
    
    # Define Loss for HmcNet.
    criterion = HmcLMLPLoss(l2_lambda=args.l2_lambda,device=device)
              
    
    
    # Create Training and Validation Dataset
    datasets = []
    for i in range(len(num_classes_list)):  
        training_dataset = HmcLMLPDataset(args.train_file, args.hierarchy_file, image_dir,level=i+1)
        datasets.append(training_dataset)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.hyperparameter_search:
        path_to_model = f'runs/hyperparameter_search_{args.dataset_name}/hmc_lmlp_{timestamp}'
    else:
        path_to_model = f'runs/hmc_lmlp_{args.dataset_name}_{timestamp}'
    
    
    # Define Trainer for HmcNet
    trainer = HmcLMLPTrainer(model=model,criterion=criterion,optimizer=optimizer,scheduler=scheduler,training_datasets=datasets,pcp_hierarchy=explicit_hierarchy,num_classes_list=num_classes_list,path_to_model=path_to_model,args=args,device=device)
    
    # Save Model ConfigParameters
    args_dict = vars(args)
    os.makedirs(path_to_model, exist_ok=True)
    with open(os.path.join(path_to_model,'model_config.json'),'w') as json_file:
        json.dump(args_dict, json_file,indent=4)
    
    if args.is_k_crossfold_val:
        trainer.train_and_validate_k_crossfold(k_folds=args.k_folds)
    else:
        trainer.train_and_validate()


def get_random_hyperparameter(base_args):
    attention_dim = random.choice([200,400])
    fc_dim = random.choice([128,256,512])
    highway_fc_dim = random.choice([128,256,512])
    highway_num_layers = random.choice([1,2])
    backbone_fc_dim = random.choice([128,256,512])
    batch_size = random.choice([128])
    learning_rate = random.choice([0.001])
    optimizer = random.choice(['adam'])
    
    print(f'Attention-Dim: {attention_dim}\n'
          f'FC-Dim: {fc_dim}\n'
          f'Highway-FC-Dim: {highway_fc_dim}\n'
          f'Highway-Num-Layers: {highway_num_layers}\n'
          f'Backbone-FC-Dim: {backbone_fc_dim}\n'
          f'Batch-Size: {batch_size}\n'
          f'Learning Rate: {learning_rate}\n'
          f'Optimizer: {optimizer}\n')
    base_args.attention_dim = attention_dim
    base_args.backbone_fc_dim = backbone_fc_dim
    base_args.fc_dim = fc_dim
    base_args.highway_fc_dim = highway_fc_dim
    base_args.highway_num_layers = highway_num_layers
    base_args.batch_size = batch_size
    base_args.learning_rate = learning_rate
    base_args.optimizer = optimizer
    return base_args




if __name__ == '__main__':
    args = parser.hmc_lmlp_parameter_parser()
    if not args.hyperparameter_search:
        # Normal Trainingloop with specific args.
        train_hmc_lmlp(args=args)
    else:
        # Hyperparameter search Trainingloop with specific base args.
        for i in range(args.num_hyperparameter_search):
            train_hmc_lmlp(args=get_random_hyperparameter(args))