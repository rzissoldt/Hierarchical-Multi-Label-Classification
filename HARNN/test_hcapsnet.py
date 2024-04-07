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
from HARNN.model.hcapsnet_model import HCapsNet
from HARNN.dataset.hcapsnet_dataset import HCapsNetDataset
from HARNN.tester.hcapsnet_tester import HCapsNetTester
from HARNN.summarywriter_evaluator import analyze_summarywriter_dir
import warnings

# Ignore specific warning types
warnings.filterwarnings("ignore", category=UserWarning)

def test_hcapsnet(args):
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
    
    # Evaluate best model.
    best_model_file_path, best_model_config = analyze_summarywriter_dir(args.hyperparameter_dir)
    # Split the path by "/"
    path_parts = best_model_file_path.split("/")

    # Navigate two folders upwards
    path_to_model = "/".join(path_parts[:-3])
    # Create Training and Validation Dataset
    test_dataset = HCapsNetDataset(annotation_file_path=args.test_file, hierarchy_file_path=args.hierarchy_file,path_to_model=path_to_model,image_dir=image_dir,target_shape=best_model_config.target_shape,hierarchy_dicts_file_path=args.hierarchy_dicts_file)
    test_dataset.is_training = False
    
    
    os.makedirs(args.path_to_results,exist_ok=True)
    

     
    # Load Input Data
    hierarchy_dicts = test_dataset.filtered_hierarchy_dicts[:best_model_config.hierarchy_depth]
    num_classes_list = dh.get_num_classes_from_hierarchy(hierarchy_dicts)[:best_model_config.hierarchy_depth]
    explicit_hierarchy = torch.tensor(dh.generate_hierarchy_matrix_from_tree(hierarchy_dicts)).to(device=device)
    
    
    
    total_class_num = sum(num_classes_list)
    
    
    # Define Model
    model = HCapsNet(output_dim=total_class_num,R=explicit_hierarchy, args=best_model_config).to(device=device)
    model_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model Parameter Count:{model_param_count}')
    print(f'Total Classes: {sum(num_classes_list)}')
    print(f'Num Classes List: {num_classes_list}')
    
    # Load Best Model Params
    best_checkpoint = torch.load(best_model_file_path)
    model.load_state_dict(best_checkpoint)
        
    
    
        
    # Define Trainer for HmcNet
    tester = HCapsNetTester(model=model,test_dataset=test_dataset,path_to_results=args.path_to_results,num_classes_list=num_classes_list,explicit_hierarchy=explicit_hierarchy,args=args,device=device)
    
    tester.test()
if __name__ == '__main__':
    args = parser.hcapsnet_parameter_parser()
    test_hcapsnet(args=args)
    