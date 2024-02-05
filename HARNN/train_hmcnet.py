
# -*- coding:utf-8 -*-
__author__ = 'Ruben'
import numpy as np
import os
import sys
import time
import logging
import torch, copy
import math
import torch.nn.functional as F
import torch.optim as optim
# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from torchvision import transforms

from torch.utils.data import DataLoader
from hmcnet_dataset import HmcNetDataset
from hmcnet_trainer import HmcNetTrainer
# Add the parent directory to the Python path
sys.path.append('../')
from utils import xtree_utils as xtree
from utils import data_helpers as dh
from utils import param_parser as parser
from hmcnet_model import HmcNet, HmcNetLoss


import warnings

# Ignore specific warning types
warnings.filterwarnings("ignore", category=UserWarning)

def train_hmcnet():
    # Define the augmentation pipeline
    args = parser.hmcnet_parameter_parser()
    
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
    model = HmcNet(feature_dim=args.feature_dim_backbone,attention_unit_size=args.attention_dim,fc_hidden_size=args.fc_dim,num_classes_list=num_classes_list,total_classes=total_classes,freeze_backbone=args.freeze_backbone,device=device)
    model_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model Parameter Count:{model_param_count}')
    
    # Define Optimzer and Scheduler    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_rate)
    model.eval().to(device)
    
    # Define Loss for HmcNet.
    criterion = HmcNetLoss(l2_lambda=args.l2_lambda,beta=args.beta,model=model,explicit_hierarchy=explicit_hierarchy,device=device)
              
    # Define the transformation pipeline for image preprocessing.
    transform = transforms.Compose([
        transforms.Resize((256, 256)),                  # Resize image to 256x256
        transforms.CenterCrop(224),                      # Center crop to 224x224 (required input size for ResNet50)
        transforms.ToTensor(),                           # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], # Normalize image using ImageNet mean and standard deviation
                         std=[0.229, 0.224, 0.225])
    ])
    
    # Create Training and Validation Dataset
    training_dataset = HmcNetDataset(args.train_file, args.hierarchy_file, image_dir,transform=transform)
    validation_dataset = HmcNetDataset(args.validation_file, args.hierarchy_file, image_dir,transform=transform)
    
    # Define Trainer for HmcNet
    trainer = HmcNetTrainer(model=model,criterion=criterion,optimizer=optimizer,scheduler=scheduler,explicit_hierarchy=explicit_hierarchy,args=args,device=device)
    
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
    tb_writer = SummaryWriter('runs/hmc_net{}'.format(timestamp))
    epoch_number = 0
    
    EPOCHS = args.epochs
    
    best_vloss = 1_000_000.
    for epoch in range(EPOCHS):
        
        avg_train_loss = trainer.train(training_loader=training_loader,epoch_index=epoch,tb_writer=tb_writer)
        avg_val_loss = trainer.validate(validation_loader=validation_loader,epoch_index=epoch,tb_writer=tb_writer)
        trainer
        tb_writer.flush()

        # Track best performance, and save the model's state
        if avg_val_loss < best_vloss:
            best_vloss = avg_val_loss
            model_path = os.path.join(path_to_model,'models','hmcnet_{}'.format(epoch_number))
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    train_hmcnet()
    
    