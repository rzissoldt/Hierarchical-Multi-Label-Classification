
# -*- coding:utf-8 -*-
__author__ = 'Ruben'
import os,json,random
import sys
import torch
from torchsummary import summary
import torch.optim as optim
# PyTorch TensorBoard support

from datetime import datetime

# Add the parent directory to the Python path
sys.path.append('../')
from utils import xtree_utils as xtree
from utils import data_helpers as dh
from utils import param_parser as parser
from HARNN.model.hmcnet_model import HmcNet, HmcNetLoss
from HARNN.dataset.hmcnet_dataset import HmcNetDataset
from HARNN.trainer.hmcnet_trainer import HmcNetTrainer

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
    image_dir = args.image_dir
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.hyperparameter_search:
        path_to_model = f'runs/hyperparameter_search_{args.dataset_name}_hierarchy_depth_{args.hierarchy_depth}_image_count_threshold_{args.image_count_threshold}_hmcnet/hmc_net_{timestamp}'
    else:
        path_to_model = f'runs/hmc_net_{args.dataset_name}_hierarchy_depth_{args.hierarchy_depth}_{timestamp}_image_count_threshold_{args.image_count_threshold}'
    # Create Training and Validation Dataset
    training_dataset = HmcNetDataset(annotation_file_path=args.train_file, hierarchy_file_path=args.hierarchy_file, path_to_model=path_to_model,hierarchy_depth=args.hierarchy_depth, image_dir=image_dir,image_count_threshold=args.image_count_threshold)
    print('Trainset Size:',len(training_dataset))
    
    hierarchy_dicts = training_dataset.filtered_hierarchy_dicts
    num_classes_list = training_dataset.num_classes_list
    explicit_hierarchy = torch.tensor(dh.generate_hierarchy_matrix_from_tree(hierarchy_dicts)).to(device=device)
    
    
    total_classes = sum(num_classes_list)

    # Define Model 
    model = HmcNet(global_average_pooling_active=args.is_backbone_global_average_pooling_active,feature_dim=args.feature_dim_backbone,is_backbone_embedding_active=args.is_backbone_embedding_active,backbone_fc_hidden_size=args.backbone_fc_dim,highway_num_layers=args.highway_num_layers,attention_unit_size=args.attention_dim,highway_fc_hidden_size=args.highway_fc_dim,fc_hidden_size=args.fc_dim,num_classes_list=num_classes_list,total_classes=total_classes,freeze_backbone=args.freeze_backbone,device=device).to(device)
    model_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model Parameter Count:{model_param_count}')
    print(f'Total Classes: {sum(num_classes_list)}')
    print(f'Num Classes List: {num_classes_list}')
    # Define Optimzer and Scheduler
    if args.optimizer == 'adam':    
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_lambda)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_lambda,nesterov=True,momentum=0.9,dampening=0.0)
    else:
        print(f'{args.optimizer} is not a valid optimizer. Quit Program.')
        return
    
    #scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=args.decay_rate,step_size=args.decay_steps)
    T_0 = 10
    T_mult = 2
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
    model.eval().to(device)
    
    # Define Loss for HmcNet.
    criterion = HmcNetLoss(l2_lambda=args.l2_lambda,beta=args.beta,explicit_hierarchy=explicit_hierarchy,device=device)
              
    
    
    
    
    
    # Define Trainer for HmcNet
    trainer = HmcNetTrainer(model=model,criterion=criterion,optimizer=optimizer,scheduler=scheduler,training_dataset=training_dataset,path_to_model=path_to_model,explicit_hierarchy=explicit_hierarchy,args=args,device=device,num_classes_list=num_classes_list)
    
    # Save Model ConfigParameters
    args_dict = vars(args)
    os.makedirs(path_to_model, exist_ok=True)
    with open(os.path.join(path_to_model,'model_config.json'),'w') as json_file:
        json.dump(args_dict, json_file,indent=4)
    with open(os.path.join(path_to_model, 'hierarchy_dicts.json'),'w') as json_file:
        json.dump(hierarchy_dicts, json_file,indent=4)
    if args.is_k_crossfold_val:
        trainer.train_and_validate_k_crossfold(k_folds=args.k_folds)
    else:
        trainer.train_and_validate()


def get_random_hyperparameter(base_args):
    attention_dim = random.choice([200,400,800])
    fc_dim = random.choice([256,512,1024])
    highway_fc_dim = random.choice([256,512,1024])
    highway_num_layers = random.choice([1,2])
    backbone_fc_dim = random.choice([128,256,512])
    batch_size = random.choice([128])
    learning_rate = random.choice([0.1])
    optimizer = random.choice(['sgd'])
    
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
    args = parser.hmcnet_parameter_parser()
    if not args.hyperparameter_search:
        # Normal Trainingloop with specific args.
        train_hmcnet(args=args)
    else:
        # Hyperparameter search Trainingloop with specific base args.
        for i in range(args.num_hyperparameter_search):
            train_hmcnet(args=get_random_hyperparameter(args))
    
    