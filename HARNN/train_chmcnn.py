# -*- coding:utf-8 -*-
__author__ = 'Ruben'
import os,json,random
import sys
import torch
import torch.optim as optim
# PyTorch TensorBoard support

from datetime import datetime

# Add the parent directory to the Python path
sys.path.append('../')
from utils import xtree_utils as xtree
from utils import data_helpers as dh
from utils import param_parser as parser
from HARNN.model.chmcnn_model import ConstrainedFFNNModel,ConstrainedFFNNModelLoss
from HARNN.dataset.chmcnn_dataset import CHMCNNDataset
from HARNN.trainer.chmcnn_trainer import CHMCNNTrainer

import warnings

# Ignore specific warning types
warnings.filterwarnings("ignore", category=UserWarning)



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
    image_dir = args.image_dir
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.hyperparameter_search:
        path_to_model = f'runs/hyperparameter_search_{args.dataset_name}_hierarchy_depth_{args.hierarchy_depth}_image_count_threshold_{args.image_count_threshold}_chmcnn/chmcnn_{timestamp}'
    else:
        path_to_model = f'runs/chmcnn_{args.dataset_name}_hierarchy_depth_{args.hierarchy_depth}_image_count_threshold_{args.image_count_threshold}_{timestamp}'
    # Create Training and Validation Dataset
    training_dataset = CHMCNNDataset(annotation_file_path=args.train_file,path_to_model=path_to_model, hierarchy_file_path=args.hierarchy_file,hierarchy_depth=args.hierarchy_depth,image_dir=image_dir,image_count_threshold=args.image_count_threshold)
    print('Trainset Size:',len(training_dataset))
    
    
    hierarchy_dicts = training_dataset.filtered_hierarchy_dicts
    num_classes_list = training_dataset.num_classes_list
    explicit_hierarchy = torch.tensor(dh.generate_hierarchy_matrix_from_tree(hierarchy_dicts)).to(device=device)
    
    
    
    total_class_num = sum(num_classes_list)
    
    
    # Define Model
    model = ConstrainedFFNNModel(output_dim=total_class_num,R=explicit_hierarchy, args=args)
    model_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model Parameter Count:{model_param_count}')

    
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
    
    # Define Loss for CHMCNN
    criterion = ConstrainedFFNNModelLoss(l2_lambda=args.l2_lambda,device=device)
    
    
    
    
    # Define Trainer for HmcNet
    trainer = CHMCNNTrainer(model=model,criterion=criterion,optimizer=optimizer,scheduler=scheduler,training_dataset=training_dataset,path_to_model=path_to_model,num_classes_list=num_classes_list,explicit_hierarchy=explicit_hierarchy,args=args,device=device)
    
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
    fc_dim = random.choice([256,512,1024,2048])
    learning_rate = random.choice([0.1])
    optimizer = random.choice(['sgd'])
    num_layers = random.choice([2,3])
    dropout_rate = random.choice([0.3,0.5,0.7])
    is_batchnorm_active = random.choice([True])
    activation_func = random.choice(['relu'])
    
    print(f'Num-Layers: {num_layers}\n'
          f'FC-Dim: {fc_dim}\n'
          f'Is Batchnorm Active: {is_batchnorm_active}\n'
          f'Dropout Rate: {dropout_rate}\n'
          f'Activation Func: {activation_func}\n'
          f'Learning Rate: {learning_rate}\n'
          f'Optimizer: {optimizer}\n')
    base_args.num_layers = num_layers
    base_args.fc_dim = fc_dim
    base_args.is_batchnorm_active = is_batchnorm_active
    base_args.activation_func = activation_func
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
            

