
import os,sys
# Add the parent directory to the Python path
sys.path.append('../')
from utils import xtree_utils as xtree
from PIL import Image
import numpy as np
import json
from skimage import color
from torch.utils.data import Dataset
import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')
from torchvision import transforms
from dataset.hierarchy_dataset import HierarchyDataset
class CHMCNNDataset(HierarchyDataset):
    def __init__(self, annotation_file_path,path_to_model, hierarchy_file_path, image_dir, image_count_threshold=0,hierarchy_dicts_file_path=None,hierarchy_depth=-1):
        super(CHMCNNDataset,self).__init__(annotation_file_path=annotation_file_path,path_to_model=path_to_model, hierarchy_file_path=hierarchy_file_path, hierarchy_depth=hierarchy_depth, image_dir=image_dir, image_count_threshold=image_count_threshold,hierarchy_dicts_file_path=hierarchy_dicts_file_path)

    def __getitem__(self, index):
        # You can customize the behavior here if needed
        return super().__getitem__(index)
