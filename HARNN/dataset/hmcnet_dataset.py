
import os,sys
# Add the parent directory to the Python path
sys.path.append('../')
from utils import xtree_utils as xtree
from PIL import Image
import numpy as np
import json
from skimage import color
from torch.utils.data import Dataset
import cv2
import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')
from torchvision import transforms
from dataset.hierarchy_dataset import HierarchyDataset
class HmcNetDataset(HierarchyDataset):
    def __init__(self, annotation_file_path, hierarchy_file_path, image_dir,path_to_model, image_count_threshold=0,hierarchy_dicts_file_path=None,hierarchy_depth=-1):
        super(HmcNetDataset,self).__init__(annotation_file_path=annotation_file_path,path_to_model=path_to_model, hierarchy_file_path=hierarchy_file_path, hierarchy_depth=hierarchy_depth, image_dir=image_dir, image_count_threshold=image_count_threshold,hierarchy_dicts_file_path=hierarchy_dicts_file_path)

    def __getitem__(self, idx):
        img_path = self.image_label_tuple_list[idx][0]
        image = Image.open(img_path)

        # Convert to RGB if it isn't already
        if image.mode != 'RGB':
            image = image.convert('RGB')

        pil_image = image  # Initialize pil_image with the original image

        if self.is_training:
            pil_image = self.train_transform(image)
        else:
            pil_image = self.validation_transform(image)
        labels = self.image_label_tuple_list[idx][1:]
        image.close()
        return pil_image, labels