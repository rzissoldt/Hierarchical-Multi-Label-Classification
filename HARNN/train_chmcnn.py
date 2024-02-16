# -*- coding:utf-8 -*-
__author__ = 'Ruben'
import numpy as np
import os,json,random
import sys
import time
import logging
import torch, copy
import torch.nn as nn
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
from HARNN.model.chmcnn_model import ConstrainedFFNNModel
from HARNN.dataset.chmcnn_dataset import CHMCNNDataset
from HARNN.trainer.chmcnn_trainer import CHMCNNTrainer

import warnings

# Ignore specific warning types
warnings.filterwarnings("ignore", category=UserWarning)

def generate_hierarchy_matrix_from_tree(hierarchy_tree):
    hierarchy_dicts = xtree.generate_dicts_per_level(hierarchy_tree)
    total_hierarchy_dict =  {}
    counter = 0 
    for hierarchy_dict in hierarchy_dicts:
        for key in hierarchy_dict.keys():
            total_hierarchy_dict[key] = counter
            counter+=1   

    hierarchy_matrix = np.zeros((len(total_hierarchy_dict),len(total_hierarchy_dict)))
    for key_parent,value_parent in total_hierarchy_dict.items():
        for key_child,value_child in total_hierarchy_dict.items():
            if key_parent == key_child:
                hierarchy_matrix[total_hierarchy_dict[key_parent],total_hierarchy_dict[key_parent]] = 1
            elif key_child.startswith(key_parent):
                hierarchy_matrix[total_hierarchy_dict[key_child],total_hierarchy_dict[key_parent]] = 1
    
    return hierarchy_matrix   

def train_chmcnn(args):
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
    explicit_hierarchy = generate_hierarchy_matrix_from_tree(hierarchy)
    
    explicit_hierarchy = torch.tensor(explicit_hierarchy)
    explicit_hierarchy = explicit_hierarchy.transpose(1,0)
    explicit_hierarchy = explicit_hierarchy.unsqueeze(0).to(device)
    
    image_dir = args.image_dir
    total_class_num = explicit_hierarchy.shape[0]
    
    
    # Define Model
    model = ConstrainedFFNNModel(output_dim=total_class_num,R=explicit_hierarchy, args=args)
    model_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model Parameter Count:{model_param_count}')
    
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
    
    # Define Loss for CHMCNN
    criterion = nn.BCELoss()
    
    # Create Training and Validation Dataset
    training_dataset = CHMCNNDataset(args.train_file, args.hierarchy_file,image_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.hyperparameter_search:
        path_to_model = f'runs/hyperparameter_search_{args.dataset_name}/chmcnn_{timestamp}'
    else:
        path_to_model = f'runs/chmcnn_{args.dataset_name}_{timestamp}'
    
    # Define Trainer for HmcNet
    trainer = CHMCNNTrainer(model=model,criterion=criterion,optimizer=optimizer,scheduler=scheduler,training_dataset=training_dataset,path_to_model=path_to_model,total_class_num=total_class_num,explicit_hierarchy=explicit_hierarchy,args=args,device=device)
    
    # Save Model ConfigParameters
    args_dict = vars(args)
    with open(os.path.join(path_to_model,'model_config.json'),'w') as json_file:
        json.dump(args_dict, json_file,indent=4)
    
    trainer.train_and_validate()

def get_random_hyperparameter(base_args):
    fc_dim = random.choice([128,256,512])
    num_layers = random.choise([1,2,3,4])
    batch_size = random.choice([128,256])
    learning_rate = random.choice([0.001])
    optimizer = random.choice(['adam'])
    
    print(f'FC-Dim: {fc_dim}\n'
          f'Num of Hidden Layers: {num_layers}\n'
          f'Batch-Size: {batch_size}\n'
          f'Learning Rate: {learning_rate}\n'
          f'Optimizer: {optimizer}\n')
    
    base_args.fc_dim = fc_dim
    base_args.num_layers = num_layers
    base_args.batch_size = batch_size
    base_args.learning_rate = learning_rate
    base_args.optimizer = optimizer
    return base_args

if __name__ == '__main__':
    args = parser.chmcnn_parameter_parser()
    if not args.hyperparameter_search:
        # Normal Trainingloop with specific args.
        train_chmcnn(args=args)
    else:
        # Hyperparameter search Trainingloop with specific base args.
        for i in range(args.num_hyperparameter_search):
            train_chmcnn(args=get_random_hyperparameter(args))