# -*- coding:utf-8 -*-
__author__ = 'Ruben'
import sys, os
import torch

# PyTorch TensorBoard support

from datetime import datetime

# Add the parent directory to the Python path
sys.path.append('../')
from utils import xtree_utils as xtree
from utils import data_helpers as dh
from utils import param_parser as parser
from HARNN.model.hmcnet_model import HmcNet
from HARNN.dataset.hmcnet_dataset import HmcNetDataset
from HARNN.tester.hmcnet_tester import HmcNetTester
from HARNN.summarywriter_evaluator import analyze_summarywriter_dir
import warnings

# Ignore specific warning types
warnings.filterwarnings("ignore", category=UserWarning)

def test_hmcnet(args):
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
   
    # Evaluate best model.
    best_model_file_path, best_model_config = analyze_summarywriter_dir(args.hyperparameter_dir)
    best_model_file_name = os.path.basename(best_model_file_path)
    os.makedirs(args.path_to_results)
    # Split the filename using '_' as the delimiter
    parts = best_model_file_name.split('_')

     
    # Load Input Data
    hierarchy = xtree.load_xtree_json(args.hierarchy_file)
    hierarchy_dicts = xtree.generate_dicts_per_level(hierarchy)
    num_classes_list = dh.get_num_classes_from_hierarchy(hierarchy_dicts)
    explicit_hierarchy = torch.tensor(dh.generate_hierarchy_matrix_from_tree(hierarchy)).to(device=device)
    
    
    image_dir = args.image_dir
    total_class_num = sum(num_classes_list)
    
    
    # Define Model
    model = HmcNet(feature_dim=args.feature_dim_backbone,attention_unit_size=args.attention_dim,backbone_fc_hidden_size=args.backbone_dim,fc_hidden_size=args.fc_dim,freeze_backbone=True,highway_fc_hidden_size=args.highway_fc_dim,highway_num_layers=args.highway_num_layers,num_classes_list=num_classes_list,total_classes=total_class_num,l2_reg_lambda=args.l2_lambda,dropout_keep_prob=args.dropout_rate,alpha=args.alpha,beta=args.beta).to(device=device)
    model_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model Parameter Count:{model_param_count}')
    print(f'Total Classes: {sum(num_classes_list)}')
    print(f'Num Classes List: {num_classes_list}')
    
    # Load Best Model Params
    best_checkpoint = torch.load(best_model_file_path)
    model.load_state_dict(best_checkpoint)
        
    # Create Training and Validation Dataset
    test_dataset = HmcNetDataset(args.test_file, args.hierarchy_file,image_dir)
    test_dataset.is_training = False
        
    # Define Trainer for HmcNet
    tester = HmcNetTester(model=model,test_dataset=test_dataset,path_to_results=args.path_to_results,num_classes_list=num_classes_list,explicit_hierarchy=explicit_hierarchy,args=args,device=device)
    
    tester.test()
if __name__ == '__main__':
    args = parser.hmcnet_parameter_parser()
    if not args.hyperparameter_search:
        # Normal Trainingloop with specific args.
        test_hmcnet(args=args)
    