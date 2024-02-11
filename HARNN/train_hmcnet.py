
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
from HARNN.model.hmcnet_model import HmcNet, HmcNetLoss
from HARNN.dataset.hmcnet_dataset import HmcNetDataset
from HARNN.trainer.hmcnet_trainer import HmcNetTrainer
import torch.nn as nn

import warnings

# Ignore specific warning types
warnings.filterwarnings("ignore", category=UserWarning)

def train_hmcnet(args):        
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
    total_classes = sum(num_classes_list)

    # Define Model 
    model = HmcNet(feature_dim=args.feature_dim_backbone,attention_unit_size=args.attention_dim,fc_hidden_size=args.fc_dim,highway_num_layers=args.highway_num_layers,highway_fc_hidden_size=args.highway_fc_dim,num_classes_list=num_classes_list,total_classes=total_classes,freeze_backbone=args.freeze_backbone,device=device).to(device)
    model_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model Parameter Count:{model_param_count}')
    
    # Define Optimzer and Scheduler
    if args.optimizer is 'adam':    
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer is 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    else:
        print(f'{args.optimizer} is not a valid optimizer. Quit Program.')
        return
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_rate)
    model.eval().to(device)
    
    # Define Loss for HmcNet.
    criterion = HmcNetLoss(l2_lambda=args.l2_lambda,beta=args.beta,model=model,explicit_hierarchy=explicit_hierarchy,device=device)
              
    # Define the transformation pipeline for image preprocessing.
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),                    
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), 
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10), 
        transforms.RandomHorizontalFlip(),                 
        transforms.RandomRotation(degrees=30),            
        transforms.CenterCrop(224),                        
        transforms.ToTensor(),                             
        transforms.Normalize(mean=[0.485, 0.456, 0.406],   
                         std=[0.229, 0.224, 0.225])
    ])
    validation_transform = transforms.Compose([
        transforms.Resize((256, 256)),                    
        transforms.CenterCrop(224),                       
        transforms.ToTensor(),                             
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                         std=[0.229, 0.224, 0.225])
    ])
    
    # Create Training and Validation Dataset
    training_dataset = HmcNetDataset(args.train_file, args.hierarchy_file, image_dir,transform=train_transform)
    validation_dataset = HmcNetDataset(args.validation_file, args.hierarchy_file, image_dir,transform=validation_transform)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.hyperparameter_search:
        path_to_model = f'runs/hyperparameter_search_{args.dataset_name}/hmc_net_{timestamp}'
    else:
        path_to_model = f'runs/hmc_net_{args.dataset_name}_{timestamp}'
    
    
    # Define Trainer for HmcNet
    trainer = HmcNetTrainer(model=model,criterion=criterion,optimizer=optimizer,scheduler=scheduler,training_dataset=training_dataset,validation_dataset=validation_dataset,path_to_model=path_to_model,explicit_hierarchy=explicit_hierarchy,args=args,device=device,num_classes_list=num_classes_list)
    
    # Save Model ConfigParameters
    args_dict = vars(args)
    with open(os.path.join(path_to_model,'model_config.json'),'w') as json_file:
        json.dump(args_dict, json_file,indent=4)
    
    trainer.train_and_validate()



def get_random_hyperparameter(base_args):
    attention_dim = [100,200,400,800]
    fc_dim = [256,512,1024,2048]
    highway_fc_dim = [256,512,1024]
    highway_num_layers = [1,2]
    batch_size = [32,64,128]
    learning_rate = [0.01,0.001]
    optimizer = ['adam','sgd']
    base_args.attention_dim = random.choice(attention_dim)
    base_args.highway_fc_dim = random.choice(highway_fc_dim)
    base_args.highway_num_layers = random.choice(highway_num_layers)
    base_args.batch_size = random.choice(batch_size)
    base_args.fc_dim = random.choice(fc_dim)
    base_args.fc_dim = random.choice(fc_dim)
    base_args.learning_rate = random.choice(learning_rate)
    base_args.optimizer = random.choice(optimizer)
    print(f'Attention-Dim: {attention_dim}'
          f'FC-Dim: {fc_dim}'
          f'Highway-FC-Dim: {highway_fc_dim}'
          f'Highway-Num-Layers: {highway_num_layers}'
          f'Batch-Size: {batch_size}'
          f'Learning Rate: {learning_rate}'
          f'Optimizer: {optimizer}')
    return base_args




if __name__ == '__main__':
    args = parser.hmcnet_parameter_parser()
    if not args.hyperparameter_search:
        # Normal Trainingloop with specific args.
        train_hmcnet(args=args)
    else:
        # Hyperparameter search Trainingloop with specific base args.
        for i in range(args.num_hyperparameter_search):
            train_hmcnet(args=get_random_hyperparameter(args))
    
    