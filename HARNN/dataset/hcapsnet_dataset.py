
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

     
class HCapsNetDataset(Dataset):
    def __init__(self, annotation_file_path, hierarchy_file_path, image_dir,target_shape):
        super(HCapsNetDataset, self).__init__()
        #self.lock = threading.Lock()
        with open(annotation_file_path,'r') as infile:
            self.image_dict = json.load(infile)
        self.hierarchy_dicts = xtree.generate_dicts_per_level(xtree.load_xtree_json(hierarchy_file_path))
        self.image_dir = image_dir
        self.pil_image_transform = transforms.ToPILImage()
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
        self.recon_transform = transforms.Compose([
            transforms.Resize((target_shape[0], target_shape[0])),
            
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
            transforms.Grayscale(num_output_channels = 1)
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
        for file_name in self.image_dict.keys():
            data_tuple = []
            labels = self.image_dict[file_name]        
            label_dict = self._find_labels_in_hierarchy_dicts(labels)
            total_class_labels = self._calc_total_class_labels(label_dict)
            
            if len(total_class_labels) == 0:
                continue
            
            data_tuple.append(os.path.join(image_dir,file_name))
        
            level = 0
            for key,labels in label_dict.items():
                data_tuple.append(torch.tensor(self._create_onehot_labels(labels,len(self.hierarchy_dicts[level])),dtype=torch.float32))
                    
                level+=1
            self.image_label_tuple_list.append(data_tuple)
        

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
        
        
        recon_image = self.recon_transform(self.pil_image_transform(pil_image))
        labels = self.image_label_tuple_list[idx][1:]
        image.close()
        return pil_image,recon_image, labels

    
    def _find_labels_in_hierarchy_dicts(self,labels):
        for label in labels:
            label_dict = {}
            labels_index = []
            level = 0
            for dict in self.hierarchy_dicts:
                labels_index = []
                label_dict['layer-{0}'.format(level)] = []
                level +=1
            level = 0
            for dict in self.hierarchy_dicts:
                labels_index = []
                for key in dict.keys():
                    if key.endswith(label):
                        labels_index.append(dict[key])
                        temp_labels = key.split('_')
                        for i in range(len(temp_labels)-2,0,-1):
                            temp_key = '_'.join(temp_labels[:i+1])
                            temp_dict = self.hierarchy_dicts[i-1]
                            temp_label = temp_dict[temp_key]
                            label_dict['layer-{0}'.format(i-1)].append(temp_label)
                            
                        
                label_dict['layer-{0}'.format(level)].extend(labels_index)
                level+=1
        
        return label_dict
    
    def _calc_total_class_labels(self,label_dict):
        level = 0
        total_class_labels = []
        for key,value in label_dict.items():
            for label in label_dict[key]:
                if level == 0:
                    total_class_labels.append(label)
                else:
                    total_class_label = label
                    for i in range(level):
                        total_class_label+=len(self.hierarchy_dicts[i].keys())
                    total_class_labels.append(total_class_label)
            level+=1
        return total_class_labels
    
    def _calc_total_classes(self):
        total_class_num = 0
        for dict in self.hierarchy_dicts:
            total_class_num+=len(dict.keys())
        return total_class_num
    
    def _create_onehot_labels(self,labels_index, num_labels):
        label = [0] * num_labels
        for item in labels_index:
            label[int(item)] = 1
        return label
