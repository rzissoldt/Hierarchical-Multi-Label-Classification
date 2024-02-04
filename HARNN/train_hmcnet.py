
# -*- coding:utf-8 -*-
__author__ = 'Ruben'

import os
import sys
import time
import logging
import torch 
import torch.nn.functional as F
import torch.optim as optim
# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from torchvision import transforms

from torch.utils.data import DataLoader
from hmcnet_dataset import HmcNetDataset
# Add the parent directory to the Python path
sys.path.append('../')
from utils import xtree_utils as xtree
from utils import data_helpers as dh
from utils import param_parser as parser
from hmcnet_model import HmcNet


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
    explicit_hierarchy_matrix = dh.generate_hierarchy_matrix_from_tree(hierarchy)
    
    image_dir = args.image_dir
    total_classes = sum(num_classes_list)

    beta = args.beta
    # Define Model and Optimzer and save them.    
    model = HmcNet(feature_dim=args.feature_dim_backbone,attention_unit_size=args.attention_dim,fc_hidden_size=args.fc_dim,num_classes_list=num_classes_list,total_classes=total_classes,freeze_backbone=args.freeze_backbone,device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    model.eval().to(device)
    
    # Define Loss for HmcNet.
    def hmcnet_loss(local_scores_list,global_logits,local_target,global_target):
        """Calculation of the HmcNet loss."""
        
        def _local_loss(local_scores_list,local_target_list):
            """Calculation of the Local loss."""
            losses = []
            for i in range(len(local_scores_list)):
                local_scores = torch.tensor([tensor.tolist() for tensor in local_scores_list[i]],dtype=torch.float32).to(device)
                local_targets = torch.transpose(torch.tensor([tensor.tolist() for tensor in local_target_list[i]],dtype=torch.float32),0,1).to(device)
                loss = F.binary_cross_entropy(local_scores, local_targets)
                mean_loss = torch.mean(loss)
                losses.append(mean_loss)
            return torch.sum(torch.tensor(losses,dtype=torch.float32).to(device))
        def _global_loss(global_logits,global_target):
            """Calculation of the Global loss."""
            global_scores = torch.sigmoid(global_logits)
            global_target = torch.transpose(torch.tensor([tensor.tolist() for tensor in global_target],dtype=torch.float32),0,1).to(device)
            loss = F.binary_cross_entropy(global_scores,global_target)
            mean_loss = torch.mean(loss).to(device)
            return mean_loss
        def _hierarchy_constraint_loss(global_logits):
            """Calculate the Hierarchy Constraint loss."""
            global_scores = torch.sigmoid(global_logits)
            hierarchy_losses = []

            for global_score in global_scores:
                temp_loss = 0.0
                for i in range(len(explicit_hierarchy_matrix)):
                    for j in range(i + 1, len(explicit_hierarchy_matrix)):
                        if explicit_hierarchy_matrix[i][j] == 1:
                            loss = beta * torch.max(torch.tensor(0.0), global_score[j] - global_score[i]) ** 2
                            temp_loss += loss
                hierarchy_losses.append(temp_loss)

            return torch.mean(torch.tensor(hierarchy_losses)).to(device)
        
        def _l2_loss(model,l2_reg_lambda):
            """Calculation of the L2-Regularization loss."""
            l2_loss = torch.tensor(0.,dtype=torch.float32).to(device)
            for param in model.parameters():
                l2_loss += torch.norm(param,p=2)**2
            return torch.tensor(l2_loss*l2_reg_lambda,dtype=torch.float32)
        print('Calc Global Loss')
        global_loss = _global_loss(global_logits=global_logits,global_target=global_target)
        print('Calc Local Loss')
        local_loss = _local_loss(local_scores_list=local_scores_list,local_target_list=local_target)
        print('Calc L2 Loss')
        l2_loss = _l2_loss(model=model,l2_reg_lambda=args.l2_lambda)
        print('Calc HIerarchy Loss')
        hierarchy_loss = _hierarchy_constraint_loss(global_logits=global_logits)
        print('Calc Total Loss')
        loss = torch.sum(torch.stack([global_loss,local_loss,l2_loss,hierarchy_loss]))
        return loss
          
    
    
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
    
    # Create Dataloader for Training and Validation Dataset
    training_loader = DataLoader(training_dataset,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers_dataloader)
    validation_loader = DataLoader(validation_dataset,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers_dataloader)
            
    num_of_train_batches = len(training_loader)
    num_of_val_batches = len(validation_loader)
    print(num_of_train_batches,num_of_val_batches)
    def train_one_epoch(epoch_index,tb_writer):
        current_loss = 0.
        last_loss = 0.
        print('Start Epoch')
        for i, data in enumerate(training_loader):
            print('Start Batch')
            # Every data instance is an input + label pair
            inputs, labels = data
            print('Start Forward Propagation')
            inputs = inputs.to(device)
            print('End Forward Propagation')
            y_total_onehot = labels[0]
            y_local_onehots = labels[1:]
            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            _, local_scores_list, global_logits = model(inputs)

            # Compute the loss and its gradients
            print('Start Calc Loss')
            loss = hmcnet_loss(local_scores_list=local_scores_list,global_logits=global_logits,local_target=y_local_onehots,global_target=y_total_onehot)
            print('Start Backpropagtion Loss')
            loss.backward()

            # Adjust learning weights
            optimizer.step()
            print('End Backpropagtion Loss')
            # Gather data and report
            current_loss += loss.item()
            if i % 100 == 99 or i % num_of_train_batches == 0:
                last_loss = current_loss / (i+1) # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(training_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                current_loss = 0.

        return last_loss
        
    
    # Initialize Tensorboard Summary Writer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0
    
    EPOCHS = args.epochs
    
    best_vloss = 1_000_000.
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer)


        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                vinputs = vinputs.to(device)
                y_total_onehot = vlabels[0]
                y_local_onehots = vlabels[1:]
                # Make predictions for this batch
                _, local_scores_list, global_logits = model(vinputs)

                # Compute the loss and its gradients
                vloss = hmcnet_loss(local_scores_list=local_scores_list,global_logits=global_logits,local_target=y_local_onehots,global_target=y_total_onehot)
                
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1
        
if __name__ == '__main__':
    train_hmcnet()