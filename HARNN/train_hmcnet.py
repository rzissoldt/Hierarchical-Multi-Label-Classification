
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
    # Define Model 
    model = HmcNet(feature_dim=args.feature_dim_backbone,attention_unit_size=args.attention_dim,fc_hidden_size=args.fc_dim,num_classes_list=num_classes_list,total_classes=total_classes,freeze_backbone=args.freeze_backbone,device=device)
    model_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model Parameter Count:{model_param_count}')
    
    # Define Optimzer and Scheduler    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_rate)
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
    sharing_strategy = "file_system"
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)

    def set_worker_sharing_strategy(worker_id: int):
        torch.multiprocessing.set_sharing_strategy(sharing_strategy)
    # Create Dataloader for Training and Validation Dataset
    kwargs = {'num_workers': args.num_workers_dataloader, 'pin_memory': args.pin_memory} if args.gpu else {}
    training_loader = DataLoader(training_dataset,batch_size=args.batch_size,shuffle=True,worker_init_fn=set_worker_sharing_strategy,**kwargs)
    validation_loader = DataLoader(validation_dataset,batch_size=args.batch_size,shuffle=True,worker_init_fn=set_worker_sharing_strategy,**kwargs)
            
    num_of_train_batches = len(training_loader)
    num_of_val_batches = len(validation_loader)

    def train_one_epoch(epoch_index,tb_writer):
        current_loss = 0.
        last_loss = 0.
        model.train()
        for i, data in enumerate(training_loader):
            # Every data instance is an input + label pair
            inputs, labels = copy.deepcopy(data)
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
            
            # Clip gradients by global norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_ratio)
            
            # Adjust learning weights
            optimizer.step()
            # Gather data and report
            current_loss += loss.item()
            last_loss = current_loss/(i+1)
            progress_info = f"Training: Epoch [{epoch_index+1}], Batch [{i+1}/{num_of_train_batches}], Loss: {last_loss}"
            print(progress_info, end='\r')
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Training/Loss', last_loss, tb_x)
            current_loss = 0.
            
            # Update Scheduler
            if i % args.decay_steps == args.decay_steps-1:
                scheduler.step()

        return last_loss
    
    def validation_after_epoch(epoch_number,tb_writer):
        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Predict classes by threshold or topk ('ts': threshold; 'tk': topk)
        eval_counter, eval_loss = 0, 0.0
        eval_pre_tk = [0.0] * args.topK
        eval_rec_tk = [0.0] * args.topK
        eval_F1_tk = [0.0] * args.topK
        eval_pre_pcp_tk = [0.0] * args.topK
        eval_rec_pcp_tk = [0.0] * args.topK
        eval_F1_pcp_tk = [0.0] * args.topK
        true_onehot_labels = []
        predicted_onehot_scores = []
        predicted_pcp_onehot_labels_ts = []
        predicted_onehot_labels_tk = [[] for _ in range(args.topK)]
        predicted_pcp_onehot_labels_tk = [[] for _ in range(args.topK)]
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = copy.deepcopy(vdata)
                vinputs = vinputs.to(device)
                y_total_onehot = vlabels[0]
                y_local_onehots = vlabels[1:]
                # Make predictions for this batch
                scores, local_scores_list, global_logits = model(vinputs)

                # Compute the loss and its gradients
                vloss = hmcnet_loss(local_scores_list=local_scores_list,global_logits=global_logits,local_target=y_local_onehots,global_target=y_total_onehot)
                
                scores = scores.cpu().numpy()
                running_vloss += vloss.item()
                # Convert each tensor to a list of lists
                y_total_onehot_list = [total_onehot.tolist() for total_onehot in list(torch.stack(y_total_onehot).t())]
                # Prepare for calculating metrics
                for i in y_total_onehot_list:
                    true_onehot_labels.append(i)
                for j in scores:
                    predicted_onehot_scores.append(j)
                # Predict by pcp-threshold
                batch_predicted_onehot_labels_ts = \
                    dh.get_pcp_onehot_label_threshold(scores=scores,explicit_hierarchy=explicit_hierarchy_matrix,num_classes_list=num_classes_list, pcp_threshold=args.pcp_threshold)
                for k in batch_predicted_onehot_labels_ts:
                    predicted_pcp_onehot_labels_ts.append(k)
                # Predict by topK
                for top_num in range(args.topK):
                    batch_predicted_onehot_labels_tk = dh.get_onehot_label_topk(scores=scores, top_num=top_num+1)
                    for i in batch_predicted_onehot_labels_tk:
                        predicted_onehot_labels_tk[top_num].append(i)
                # Predict by pcp-topK
                for top_num in range(args.topK):
                    batch_predicted_pcp_onehot_labels_tk = dh.get_pcp_onehot_label_topk(scores=scores,explicit_hierarchy=explicit_hierarchy_matrix,pcp_threshold=args.pcp_threshold,num_classes_list=num_classes_list, top_num=top_num+1)
                    for i in batch_predicted_pcp_onehot_labels_tk:
                        predicted_pcp_onehot_labels_tk[top_num].append(i)
                
                eval_loss = running_vloss/(eval_counter+1)
                progress_info = f"Validation: Epoch [{epoch_number+1}], Batch [{eval_counter+1}/{num_of_val_batches}], Loss: {eval_loss}"
                print(progress_info, end='\r')
                eval_counter+=1
            # Calculate Precision & Recall & F1
            eval_pre_pcp_ts = precision_score(y_true=np.array(true_onehot_labels),
                                          y_pred=np.array(predicted_pcp_onehot_labels_ts), average='micro')
            eval_rec_pcp_ts = recall_score(y_true=np.array(true_onehot_labels),
                                       y_pred=np.array(predicted_pcp_onehot_labels_ts), average='micro')
            eval_F1_pcp_ts = f1_score(y_true=np.array(true_onehot_labels),
                                  y_pred=np.array(predicted_pcp_onehot_labels_ts), average='micro')
            for top_num in range(args.topK):
                eval_pre_tk[top_num] = precision_score(y_true=np.array(true_onehot_labels),
                                                       y_pred=np.array(predicted_onehot_labels_tk[top_num]),
                                                       average='micro')
                eval_rec_tk[top_num] = recall_score(y_true=np.array(true_onehot_labels),
                                                    y_pred=np.array(predicted_onehot_labels_tk[top_num]),
                                                    average='micro')
                eval_F1_tk[top_num] = f1_score(y_true=np.array(true_onehot_labels),
                                               y_pred=np.array(predicted_onehot_labels_tk[top_num]),
                                               average='micro')
            for top_num in range(args.topK):
                eval_pre_pcp_tk[top_num] = precision_score(y_true=np.array(true_onehot_labels),
                                                       y_pred=np.array(predicted_pcp_onehot_labels_tk[top_num]),
                                                       average='micro')
                eval_rec_pcp_tk[top_num] = recall_score(y_true=np.array(true_onehot_labels),
                                                    y_pred=np.array(predicted_pcp_onehot_labels_tk[top_num]),
                                                    average='micro')
                eval_F1_pcp_tk[top_num] = f1_score(y_true=np.array(true_onehot_labels),
                                               y_pred=np.array(predicted_pcp_onehot_labels_tk[top_num]),
                                               average='micro')
            # Calculate the average AUC
            eval_auc = roc_auc_score(y_true=np.array(true_onehot_labels),
                                     y_score=np.array(predicted_onehot_scores), average='micro')
            # Calculate the average PR
            eval_prc = average_precision_score(y_true=np.array(true_onehot_labels),
                                               y_score=np.array(predicted_onehot_scores), average='micro')
            tb_writer.add_scalar('Validation/Loss',eval_loss,epoch_number)
            tb_writer.add_scalar('Validation/AverageAUC',eval_auc,epoch_number)
            tb_writer.add_scalar('Validation/AveragePrecision',eval_prc,epoch_number)
            # Add each scalar individually
            for i, precision in enumerate(eval_pre_tk):
                tb_writer.add_scalar(f'Validation/PrecisionTopK/{i}', precision, global_step=epoch_number)
            for i, recall in enumerate(eval_rec_tk):
                tb_writer.add_scalar(f'Validation/RecallTopK/{i}', recall, global_step=epoch_number)
            for i, f1 in enumerate(eval_F1_tk):
                tb_writer.add_scalar(f'Validation/F1TopK/{i}', f1, global_step=epoch_number)
            
            tb_writer.add_scalar('Validation/PCPPrecision',eval_pre_pcp_ts,epoch_number)
            tb_writer.add_scalar('Validation/PCPRecall',eval_rec_pcp_ts,epoch_number)
            tb_writer.add_scalar('Validation/PCPF1',eval_F1_pcp_ts,epoch_number)
            for i, precision in enumerate(eval_pre_pcp_tk):
                tb_writer.add_scalar(f'Validation/PCPPrecisionTopK/{i}', precision, global_step=epoch_number)
            for i, recall in enumerate(eval_rec_pcp_tk):
                tb_writer.add_scalar(f'Validation/PCPRecallTopK/{i}', recall, global_step=epoch_number)
            for i, f1 in enumerate(eval_F1_pcp_tk):
                tb_writer.add_scalar(f'Validation/PCPF1TopK/{i}', f1, global_step=epoch_number)
        
            print("All Validation set: Loss {0:g} | AUC {1:g} | AUPRC {2:g}".format(eval_loss, eval_auc, eval_prc))
            # Predict by pcp
            print("Predict by PCP thresholding: PCP-Precision {0:g}, PCP-Recall {1:g}, PCP-F1 {2:g}".format(eval_pre_pcp_ts, eval_rec_pcp_ts, eval_F1_pcp_ts))
            # Predict by topK
            print("Predict by topK:")
            for top_num in range(args.topK):
                print("Top{0}: Precision {1:g}, Recall {2:g}, F1 {3:g}".format(top_num+1, eval_pre_tk[top_num], eval_rec_tk[top_num], eval_F1_tk[top_num])) 
            # Predict by PCP-topK
            print("Predict by PCP-topK:")
            for top_num in range(args.topK):
                print("Top{0}: PCP-Precision {1:g}, PCP-Recall {2:g}, PCP-F1 {3:g}".format(top_num+1, eval_pre_pcp_tk[top_num], eval_rec_pcp_tk[top_num], eval_F1_pcp_tk[top_num]))  
            return eval_loss
            
    
    # Initialize Tensorboard Summary Writer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path_to_model = 'runs/hmc_net{}'.format(timestamp)
    writer = SummaryWriter('runs/hmc_net{}'.format(timestamp))
    epoch_number = 0
    
    EPOCHS = args.epochs
    
    best_vloss = 1_000_000.
    for epoch in range(EPOCHS):
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch, writer)
        avg_vloss = validation_after_epoch(epoch,writer)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = os.path.join(path_to_model,'models','hmcnet_{}'.format(epoch_number))
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    train_hmcnet()
    
    