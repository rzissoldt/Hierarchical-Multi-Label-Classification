
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
from HARNN.model.hcapsnet_model import HCapsNet, HCapsNetLoss
from HARNN.dataset.hcapsnet_dataset import HCapsNetDataset
from HARNN.trainer.hcapsnet_trainer import HCapsNetTrainer
from torchsummary import summary
import warnings

# Ignore specific warning types
warnings.filterwarnings("ignore", category=UserWarning)




def train_hcapsnet(args):        
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
    num_classes_list = dh.get_num_classes_from_hierarchy(hierarchy_dicts,args.image_count_threshold)[:args.hierarchy_depth]
    explicit_hierarchy = torch.tensor(dh.generate_hierarchy_matrix_from_tree(hierarchy,args.hierarchy_depth,image_count_threshold=args.image_count_threshold)).to(device=device)
    
    image_dir = args.image_dir
    

    # Define Model 
    model = HCapsNet(feature_dim=None,input_shape=args.input_size,filter_list=args.filter_list,secondary_capsule_input_dim=args.secondary_capsule_input_dim,num_classes_list=num_classes_list,pcap_n_dims=8,scap_n_dims=16,fc_hidden_size=256,num_layers=2,target_shape=args.target_shape,device=device).to(device)
    model_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model Parameter Count:{model_param_count}')
    print(f'Total Classes: {sum(num_classes_list)}')
    print(f'Num Classes List: {num_classes_list}')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.numel())
    # Define Optimzer and Scheduler
    if args.optimizer == 'adam':    
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_lambda,nesterov=True,momentum=0.9,dampening=0.0)
    else:
        print(f'{args.optimizer} is not a valid optimizer. Quit Program.')
        return
    T_0 = 10
    T_mult = 2
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
    model.eval().to(device)
    
    # Define Loss for HmcNet.
    criterion = HCapsNetLoss(device=device,tau=args.tau)
              
    
    
    # Create Training and Validation Dataset
    training_dataset = HCapsNetDataset(args.train_file, args.hierarchy_file, image_dir,target_shape=args.target_shape,image_count_threshold=args.image_count_threshold)
    print('Trainset Size:',len(training_dataset))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.hyperparameter_search:
        path_to_model = f'runs/hyperparameter_search_{args.dataset_name}_hierarchy_depth_{args.hierarchy_depth}_image_count_threshold_{args.image_count_threshold}_hcapsnet/hcapsnet_{timestamp}'
    else:
        path_to_model = f'runs/hcapsnet_{args.dataset_name}_hierarchy_depth_{args.hierarchy_depth}_image_count_threshold_{args.image_count_threshold}_{timestamp}'
    
    
    # Define Trainer for HmcNet
    trainer = HCapsNetTrainer(model=model,criterion=criterion,optimizer=optimizer,scheduler=scheduler,training_dataset=training_dataset,path_to_model=path_to_model,explicit_hierarchy=explicit_hierarchy,args=args,device=device,num_classes_list=num_classes_list)
    
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


def hyperparameter_search(base_args):
    learning_rate = random.choice([0.1])
    optimizer = random.choice(['sgd'])
    fc_dim = random.choice([256])
    print(f'FC-DIM: {fc_dim}\n'
          f'Learning Rate: {learning_rate}\n'
          f'Optimizer: {optimizer}\n')
    
    base_args.learning_rate = learning_rate
    base_args.optimizer = optimizer
    filter_list = [16,32,64]
    base_args.filter_list = filter_list
    base_args.secondary_capsule_input_dim = 1152
    train_hcapsnet(args=base_args)
    filter_list = [32,64,128]
    base_args.filter_list = filter_list
    base_args.secondary_capsule_input_dim = 2304
    
    train_hcapsnet(args=base_args)
    
    
    filter_list = [64,128,256]
    base_args.filter_list = filter_list
    base_args.secondary_capsule_input_dim = 4608
    train_hcapsnet(args=base_args)
    




if __name__ == '__main__':
    args = parser.hcapsnet_parameter_parser()
    if not args.hyperparameter_search:
        # Normal Trainingloop with specific args.
        train_hcapsnet(args=args)
    else:
        hyperparameter_search(base_args=args)
            
    
    