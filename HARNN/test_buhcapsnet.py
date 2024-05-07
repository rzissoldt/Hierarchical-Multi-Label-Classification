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
from HARNN.model.buhcapsnet_model import BUHCapsNet
from HARNN.dataset.buhcapsnet_dataset import BUHCapsNetDataset
from HARNN.tester.buhcapsnet_tester import BUHCapsNetTester
from HARNN.summarywriter_evaluator import analyze_summarywriter_dir
import warnings, json
from types import SimpleNamespace
# Ignore specific warning types
warnings.filterwarnings("ignore", category=UserWarning)

def test_buhcapsnet(args):
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
    if args.hyperparameter_search:
        # Evaluate best model.
        best_model_file_path, best_model_config = analyze_summarywriter_dir(args.hyperparameter_dir)
        
        # Split the path by "/"
        path_parts = best_model_file_path.split("/")

        # Navigate two folders upwards
        path_to_model = "/".join(path_parts[:-3])
    else:
        # Evaluate best model.
        path_to_model = args.model_dir
        best_model_dir = os.path.join(path_to_model,'models')
        best_model_file_path = os.path.join(best_model_dir,os.listdir(best_model_dir)[0])
        best_model_config_path = os.path.join(path_to_model,'model_config.json')
        with open(best_model_config_path) as infile:
            best_model_config = SimpleNamespace(**json.load(infile))
    os.makedirs(args.path_to_results,exist_ok=True)
    # Create Training and Validation Dataset
    test_dataset = BUHCapsNetDataset(annotation_file_path=args.test_file, path_to_model=path_to_model,hierarchy_file_path=args.hierarchy_file,image_dir=image_dir,hierarchy_dicts_file_path=args.hierarchy_dicts_file,hierarchy_depth=best_model_config.hierarchy_depth)
    test_dataset.is_training = False
    
    

     
    # Load Input Data
    hierarchy_dicts = test_dataset.filtered_hierarchy_dicts
    num_classes_list = dh.get_num_classes_from_hierarchy(hierarchy_dicts)
    explicit_hierarchy = torch.tensor(dh.generate_hierarchy_matrix_from_tree(hierarchy_dicts)).to(device=device)
    
    
    
    total_class_num = sum(num_classes_list)
    
    
    # Define Model
    model = BUHCapsNet(pcap_n_dims=best_model_config.pcap_n_dims,scap_n_dims=best_model_config.scap_n_dims,num_classes_list=num_classes_list,routings=best_model_config.routing_iterations,args=args,device=device)
    model_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model Parameter Count:{model_param_count}')
    print(f'Total Classes: {sum(num_classes_list)}')
    print(f'Num Classes List: {num_classes_list}')
    
    # Load Best Model Params
    best_checkpoint = torch.load(best_model_file_path)
    model.load_state_dict(best_checkpoint)
        
    
    
        
    # Define Trainer for HmcNet
    tester = BUHCapsNetTester(model=model,test_dataset=test_dataset,path_to_results=args.path_to_results,num_classes_list=num_classes_list,hierarchy_dicts=hierarchy_dicts,sample_images_size=args.sample_image_count,explicit_hierarchy=explicit_hierarchy,args=args,device=device)
    
    tester.test()
if __name__ == '__main__':
    args = parser.buhcapsnet_parameter_parser()
    test_buhcapsnet(args=args)
    