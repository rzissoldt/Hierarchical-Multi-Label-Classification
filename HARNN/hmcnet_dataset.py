
import os,sys
# Add the parent directory to the Python path
sys.path.append('../')
from utils import xtree_utils as xtree
from PIL import Image
import numpy as np
import json
from skimage import color
from torch.utils.data import Dataset
import threading
class HmcNetDataset(Dataset):
    def __init__(self, annotation_file, hierarchy_file, image_dir, transform=None, target_transform=None):
        self.lock = threading.Lock()
        with open(annotation_file,'r') as infile:
            self.image_dict = json.load(infile)
        self.hierarchy_dicts = xtree.generate_dicts_per_level(xtree.load_xtree_json(hierarchy_file))
        self.image_dir = image_dir
        self.transform = transform
        self.target_transform = target_transform
        self.image_label_tuple_list = []
        for file_name in self.image_dict.keys():
            data_tuple = []
            labels = self.image_dict[file_name]        
            label_dict = self._find_labels_in_hierarchy_dicts(labels)
            total_class_labels = self._calc_total_class_labels(label_dict)
            total_class_num = self._calc_total_classes()
            if len(total_class_labels) == 0:
                continue
            
            data_tuple.append(os.path.join(image_dir,file_name))
            data_tuple.append(self._create_onehot_labels(total_class_labels, total_class_num))
            level = 0
            for key,labels in label_dict.items():
                data_tuple.append(self._create_onehot_labels(labels,len(self.hierarchy_dicts[level])))
                    
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
        if self.transform:
            image = self.transform(image)
        labels = self.image_label_tuple_list[idx][1:]
        return image, labels
    
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
