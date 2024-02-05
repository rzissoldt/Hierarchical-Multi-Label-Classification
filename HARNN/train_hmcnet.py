
# -*- coding:utf-8 -*-
__author__ = 'Ruben'
import numpy as np
import os
import sys
import time
import logging
import torch 
import math
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
    model_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model Parameter Count:{model_param_count}')
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
                mask = explicit_hierarchy_matrix == 1
                score_diff = global_score.unsqueeze(0) - global_score.unsqueeze(1)                
                loss = beta * torch.max(torch.tensor(0.0), score_diff[mask]) ** 2
                temp_loss = torch.sum(loss)
                hierarchy_losses.append(temp_loss)
            return torch.mean(torch.tensor(hierarchy_losses)).to(device)
        
        def _l2_loss(model,l2_reg_lambda):
            """Calculation of the L2-Regularization loss."""
            l2_loss = torch.tensor(0.,dtype=torch.float32).to(device)
            for param in model.parameters():
                l2_loss += torch.norm(param,p=2)**2
            return torch.tensor(l2_loss*l2_reg_lambda,dtype=torch.float32)
        global_loss = _global_loss(global_logits=global_logits,global_target=global_target)
        local_loss = _local_loss(local_scores_list=local_scores_list,local_target_list=local_target)
        l2_loss = _l2_loss(model=model,l2_reg_lambda=args.l2_lambda)
        hierarchy_loss = _hierarchy_constraint_loss(global_logits=global_logits)
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

    def train_one_epoch(epoch_index,tb_writer):
        current_loss = 0.
        last_loss = 0.
        
        for i, data in enumerate(training_loader):
            # Every data instance is an input + label pair
            inputs, labels = data
            inputs = inputs.to(device)
            y_total_onehot = labels[0]
            y_local_onehots = labels[1:]
            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            _, local_scores_list, global_logits = model(inputs)

            # Compute the loss and its gradients
            loss = hmcnet_loss(local_scores_list=local_scores_list,global_logits=global_logits,local_target=y_local_onehots,global_target=y_total_onehot)
            loss.backward()

            # Adjust learning weights
            optimizer.step()
            # Gather data and report
            current_loss += loss.item()
            
            progress_info = f"Epoch [{epoch_index}], Batch [{i+1}/{num_of_train_batches}], Loss: {current_loss}"
            print(progress_info, end='\r')
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', current_loss, tb_x)
            current_loss = 0.

        return last_loss
    
    def validation_after_epoch():
        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Predict classes by threshold or topk ('ts': threshold; 'tk': topk)
        eval_counter, eval_loss = 0, 0.0
        eval_pre_tk = [0.0] * args.topK
        eval_rec_tk = [0.0] * args.topK
        eval_F1_tk = [0.0] * args.topK

        true_onehot_labels = []
        predicted_onehot_scores = []
        predicted_onehot_labels_ts = []
        predicted_onehot_labels_tk = [[] for _ in range(args.topK)]
        
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                vinputs = vinputs.to(device)
                y_total_onehot = vlabels[0]
                y_local_onehots = vlabels[1:]
                # Make predictions for this batch
                scores, local_scores_list, global_logits = model(vinputs)

                # Compute the loss and its gradients
                vloss = hmcnet_loss(local_scores_list=local_scores_list,global_logits=global_logits,local_target=y_local_onehots,global_target=y_total_onehot)
                
                running_vloss += vloss
                # Prepare for calculating metrics
                for i in y_total_onehot:
                    true_onehot_labels.append(i)
                for j in scores:
                    predicted_onehot_scores.append(j)
                # Predict by threshold
                batch_predicted_onehot_labels_ts = \
                    dh.get_onehot_label_threshold(scores=scores, threshold=args.threshold)
                for k in batch_predicted_onehot_labels_ts:
                    predicted_onehot_labels_ts.append(k)
                # Predict by topK
                for top_num in range(args.topK):
                    batch_predicted_onehot_labels_tk = dh.get_onehot_label_topk(scores=scores, top_num=top_num+1)
                    for i in batch_predicted_onehot_labels_tk:
                        predicted_onehot_labels_tk[top_num].append(i)
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
            model_path = 'runs/model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)
    
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
        validation_after_epoch()

        

        epoch_number += 1



def weighted_path_selecting(paths, scores, threshold, hierarchy_depth):
    def _weigh(hierarchy_level):
        return 1 - (hierarchy_level/(hierarchy_depth+1))
    
    selected_paths = []
    for path in paths:
        path_score = sum([_weigh(hierachy_level) * math.log(scores[path[hierachy_level]]) for hierachy_level in range(len(path))])
        if path_score > threshold:
            selected_paths.append(path)
    return selected_paths
            
        

def dynamic_threshold_pruning(scores, explicit_hierarchy,num_classes_list):
    def _get_children(nodes,explicit_hierarchy):
        children = []
        for index in nodes:
            indices = list(np.where(explicit_hierarchy[index,:] == 1)[0])
            children.extend(indices)
        return children
    
    def _dynamic_filter(scores,mask=None):
        paths = []
        max_value = max(scores)
        threshold = max_value*0.8
        for i in range(len(scores)):
            if mask is None:
                if scores[i] > threshold:
                    paths.append(i)
            else:
                if i in mask:
                    if scores[i] > threshold:
                        paths.append(i)
        return paths
    
    def _get_paths(candidate_nodes):
        def _find_paths(node, path):
            paths = []
            hierarchy_range_lower = sum(num_classes_list[:len(path)])
            hierarchy_range_upper = hierarchy_range_lower+num_classes_list[len(path)]
            hierarchy_prev = sum(num_classes_list[:len(path)-1])
            if not (node >= hierarchy_prev and node < hierarchy_range_lower):
                return None
            children = [i for i, value in enumerate(explicit_hierarchy[node]) if node != i and value == 1 and i in candidate_nodes and (i >= hierarchy_range_lower and i < hierarchy_range_upper)]
            if not children:  # If no children, return the current path
                return [path]
            for child in children:
                paths.extend(_find_paths(child, path + [child]))
            return paths

        all_paths = []
        for node in candidate_nodes:
            paths_from_node = _find_paths(node, [node])
            if paths_from_node is None:
                continue
            all_paths.extend(paths_from_node)
        return all_paths
    
    all_nodes = []
    for h in range(len(num_classes_list)):
        if h == 0:
            selected = _dynamic_filter(scores[:num_classes_list[h]])
            all_nodes.extend(selected)
        else:
            children = _get_children(selected,explicit_hierarchy)
            temp_selected = _dynamic_filter(scores,children)
            all_nodes.extend(temp_selected)
    all_nodes = list(set(all_nodes))
    all_paths = _get_paths(all_nodes)
    
    return all_paths
            
         
            



    

if __name__ == '__main__':
    hierarchy_file = "../data/image_harnn/bauwerke/Bauwerk_tree_final_with_image_count_threshold_1000.json"
    
    hierarchy = xtree.load_xtree_json(hierarchy_file)
    hierarchy_dicts = xtree.generate_dicts_per_level(hierarchy)
    num_classes_list = dh.get_num_classes_from_hierarchy(hierarchy_dicts)
    explicit_hierarchy_matrix = dh.generate_hierarchy_matrix_from_tree(hierarchy)
    scores = list(np.random.random(size=explicit_hierarchy_matrix.shape[0]))
    paths = dynamic_threshold_pruning(scores=scores,explicit_hierarchy=explicit_hierarchy_matrix,num_classes_list=num_classes_list)
    selected_paths = weighted_path_selecting(paths=paths,scores=scores,threshold=-1,hierarchy_depth=len(num_classes_list))
    print(selected_paths)
    
    