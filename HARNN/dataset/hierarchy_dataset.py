
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
class HierarchyDataset(Dataset):
    def __init__(self, annotation_file_path, hierarchy_file_path, hierarchy_depth, image_dir,image_count_threshold):
        super(HierarchyDataset, self).__init__()
        with open(annotation_file_path,'r') as infile:
            self.image_dict = json.load(infile)
        self.hierarchy = xtree.load_xtree_json(hierarchy_file_path)
        self.hierarchy_dicts = xtree.generate_dicts_per_level(root=self.hierarchy)
        if hierarchy_depth == -1:
            self.hierarchy_depth = len(self.hierarchy_dicts)
        else:
            self.hierarchy_depth = hierarchy_depth
        self.image_dir = image_dir
        self.image_count_threshold = image_count_threshold
        self.hierarchy_depth = hierarchy_depth
        self.layer_distribution_dict = []
        self.global_hierarchy_dict = {}
        self.global_distribution_dict = {}
        self.initialize_distribution_dicts(self.hierarchy_dicts)
        # Define the transformation pipeline for image preprocessing.
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),                    
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), 
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10), 
            transforms.RandomHorizontalFlip(),                 
            transforms.RandomRotation(degrees=30),            
            transforms.CenterCrop(224),                        
            transforms.ToTensor(),                             
            transforms.Normalize(mean=[0.485, 0.456, 0.406],   
                         std=[0.229, 0.224, 0.225])
        ])
        self.validation_transform = transforms.Compose([
            transforms.Resize((256, 256)),                    
            transforms.CenterCrop(224),                       
            transforms.ToTensor(),                             
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                             std=[0.229, 0.224, 0.225])
        ])
        self.is_training = True
        self.image_label_tuple_list = []
        
        print('Image dict size:',len(self.image_dict.keys()))
        self.filtered_hierarchy_dicts = self.filter_hierarchy_dicts_with_threshold()
        if hierarchy_depth == -1:
            self.hierarchy_depth = len(self.filtered_hierarchy_dicts)
        else:
            self.hierarchy_depth = hierarchy_depth
        self.layer_distribution_dict = []
        self.global_hierarchy_dict = {}
        self.global_distribution_dict = {}
        self.initialize_distribution_dicts(self.filtered_hierarchy_dicts)
        for file_name in self.image_dict.keys():
            data_tuple = []
            labels = self.image_dict[file_name]        
            label_dict = self._find_labels_in_hierarchy_dicts(labels,self.filtered_hierarchy_dicts)
            sliced_label_dict = dict(list(label_dict.items()))
            total_class_labels = self._calc_total_class_labels(label_dict,self.filtered_hierarchy_dicts)
            total_class_num = self._calc_total_classes()
            if len(total_class_labels) == 0:
                continue
            
            data_tuple.append(os.path.join(image_dir,file_name))
            data_tuple.append(torch.tensor(self._create_onehot_labels(total_class_labels, total_class_num),dtype=torch.float32))
            level = 0
            for key,labels in sliced_label_dict.items():
                data_tuple.append(torch.tensor(self._create_onehot_labels(labels,len(self.filtered_hierarchy_dicts[level])),dtype=torch.float32))
                    
                level+=1
            self.image_label_tuple_list.append(data_tuple)
    
    def initialize_distribution_dicts(self,hierarchy_dicts):
        for hierarchy_dict in hierarchy_dicts:
            layer_dict = {}
            for key in hierarchy_dict:
                layer_dict[hierarchy_dict[key]] = 0
            self.layer_distribution_dict.append(layer_dict)
            
        counter = 0
        for hierarchy_dict in self.hierarchy_dicts:
            for key in hierarchy_dict:
                self.global_hierarchy_dict[key] = counter
                self.global_distribution_dict[counter] = 0
                counter+=1
                
    def filter_hierarchy_dicts_with_threshold(self):        
        for file_name in self.image_dict.keys():
            labels = self.image_dict[file_name]        
            label_dict = self._find_labels_in_hierarchy_dicts(labels,self.hierarchy_dicts)
            level = 0
            for layer_key in label_dict.keys():
                for label_idx in label_dict[layer_key]:
                    self.layer_distribution_dict[level][label_idx] += 1
                level+=1
            total_class_idxs = self._calc_total_class_labels(label_dict,self.hierarchy_dicts)
            for total_class_idx in total_class_idxs:
                self.global_distribution_dict[total_class_idx] +=1
        ## Generate per Layer Distribution
        level = 0
        class_distribution_dicts = []
        for layer_dict in self.hierarchy_dicts:
            layer_distribution_dict = self.layer_distribution_dict[level]
            class_distribution_dict = {}
            for label in layer_dict:
                class_distribution_dict[label] = layer_distribution_dict[layer_dict[label]]
            class_distribution_dicts.append(class_distribution_dict)
            level+=1
        global_class_distribution_dict = {}
        for label in self.global_hierarchy_dict:
            global_class_distribution_dict[label] = self.global_distribution_dict[self.global_hierarchy_dict[label]]
        
        ## Filter with threshold
        filtered_class_distribution_dicts = []
        for distribution_dict in class_distribution_dicts:
            filtered_distribution_dict = {key: value for key, value in distribution_dict.items() if value >= self.image_count_threshold}
            filtered_class_distribution_dicts.append(filtered_distribution_dict)
        filtered_global_distribution_dict = {key: value for key, value in global_class_distribution_dict.items() if value >= self.image_count_threshold}
        
        ## Generate filtered hierarchy dicts
        filtered_hierarchy_dicts = []
        counter = 0
        for filtered_class_distribution_dict in filtered_class_distribution_dicts:
            filtered_hierarchy_dict = {}
            for key in filtered_class_distribution_dict:
                filtered_hierarchy_dict[key] = counter
                counter +=1
            if len(list(filtered_hierarchy_dict.keys())) == 0:
                continue 
            filtered_hierarchy_dicts.append(filtered_hierarchy_dict)
            counter = 0
        return filtered_hierarchy_dicts[:self.hierarchy_depth]
    def _find_labels_in_hierarchy_dicts(self,labels, hierarchy_dicts):
        for label in labels:
            path = xtree.get_id_path(self.hierarchy,label)[:self.hierarchy_depth]
            
            label_dict = {}
            labels_index = []
            level = 0
            for dict in hierarchy_dicts:
                labels_index = []
                label_dict['layer-{0}'.format(level)] = []
                level +=1
            level = 0
            for i in range(1,len(path)):
                temp_key = '_'.join(path[:i+1])
                temp_dict = hierarchy_dicts[i-1]
                if temp_key not in temp_dict.keys():
                    break
                temp_label = temp_dict[temp_key]
                label_dict['layer-{0}'.format(i-1)].append(temp_label)

                            
                         
            label_dict['layer-{0}'.format(level)].extend(labels_index)
            level+=1
        
        return label_dict
    
    def _calc_total_class_labels(self,label_dict,hierarchy_dicts):
        level = 0
        total_class_labels = []
        for key,value in label_dict.items():
            for label in label_dict[key]:
                if level == 0:
                    total_class_labels.append(label)
                else:
                    total_class_label = label
                    for i in range(level):
                        total_class_label+=len(hierarchy_dicts[i].keys())
                    total_class_labels.append(total_class_label)
            level+=1
        return total_class_labels
    
    def _calc_total_classes(self):
        total_class_num = 0
        for dict in self.filtered_hierarchy_dicts:
            total_class_num+=len(dict.keys())
        return total_class_num
    
    def _create_onehot_labels(self,labels_index, num_labels):
        label = [0] * num_labels
        for item in labels_index:
            label[int(item)] = 1
        return label
    
    def __len__(self):
        return len(self.image_label_tuple_list)

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
        labels = self.image_label_tuple_list[idx][1]
        image.close()
        return pil_image, labels
   