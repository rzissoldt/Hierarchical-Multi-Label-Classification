
# -*- coding:utf-8 -*-
__author__ = 'Ruben'
import numpy as np
import os,json
import sys
import time
import logging
import torch, copy
import math
from torchsummary import summary
import torch.optim as optim
# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from torchvision import transforms

from torch.utils.data import DataLoader

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
    model = HmcNet(feature_dim=args.feature_dim_backbone,attention_unit_size=args.attention_dim,fc_hidden_size=args.fc_dim,num_classes_list=num_classes_list,total_classes=total_classes,freeze_backbone=args.freeze_backbone,device=device).to(device)
    model_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model Parameter Count:{model_param_count}')
    
    # Define Optimzer and Scheduler    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
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
    
    # Define Trainer for HmcNet
    trainer = HmcNetTrainer(model=model,criterion=criterion,optimizer=optimizer,scheduler=scheduler,explicit_hierarchy=explicit_hierarchy,args=args,device=device,num_classes_list=num_classes_list)
    
    sharing_strategy = "file_system"
    def set_worker_sharing_strategy(worker_id: int):
        torch.multiprocessing.set_sharing_strategy(sharing_strategy)
    # Create Dataloader for Training and Validation Dataset
    kwargs = {'num_workers': args.num_workers_dataloader, 'pin_memory': args.pin_memory} if args.gpu else {}
    training_loader = DataLoader(training_dataset,batch_size=args.batch_size,shuffle=True,worker_init_fn=set_worker_sharing_strategy,**kwargs)
    validation_loader = DataLoader(validation_dataset,batch_size=args.batch_size,shuffle=True,worker_init_fn=set_worker_sharing_strategy,**kwargs)  
    
    # Initialize Tensorboard Summary Writer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path_to_model = 'runs/hmc_net{}'.format(timestamp)
    tb_writer = SummaryWriter(path_to_model)
    
    # Save Model ConfigParameters
    args_dict = vars(args)
    with open(os.path.join(path_to_model,'model_config.json'),'w') as json_file:
        json.dump(args_dict, json_file,indent=4)
    
    counter = 0
    
    EPOCHS = args.epochs
    
    best_vloss = 1_000_000.
    for epoch in range(EPOCHS):
        
        avg_train_loss = trainer.train(training_loader=training_loader,epoch_index=epoch,tb_writer=tb_writer)
        
        calc_metrics = epoch == EPOCHS-1
        
        avg_val_loss = trainer.validate(validation_loader=validation_loader,epoch_index=epoch,tb_writer=tb_writer,calc_metrics=calc_metrics)
        tb_writer.flush()
        print(f'Epoch {epoch+1}: Average Train Loss {avg_train_loss}, Average Validation Loss {avg_val_loss}')
        # Track best performance, and save the model's state
        if avg_val_loss < best_vloss:
            best_vloss = avg_val_loss
            model_path = os.path.join(path_to_model,'models',f'hmcnet_{epoch}')
            counter = 0
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
        else:
            counter += 1
            if counter >= args.early_stopping_patience:
                print('Early stopping triggered.')
                avg_val_loss = trainer.validate(validation_loader=validation_loader,epoch_index=epoch,tb_writer=tb_writer,calc_metrics=True)
                #os.makedirs(os.path.dirname(model_path), exist_ok=True)
                #torch.save(model.state_dict(), model_path)
                break

if __name__ == '__main__':
    # Define the augmentation pipeline
    args = parser.hmcnet_parameter_parser()
    train_hmcnet(args=args)
    
    