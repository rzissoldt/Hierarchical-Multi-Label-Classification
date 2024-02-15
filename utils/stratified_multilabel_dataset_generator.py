import json, os, shlex
from urllib.parse import unquote, quote
from pathlib import Path
from xtree_utils import load_xtree_json,find_all_paths, get_id_paths
from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold,MultilabelStratifiedShuffleSplit
import numpy as np
def generate_train_and_validation_dataset(image_dict_file_path,xtree_file_path, output_path, image_dir=None):
    if image_dir is None:
        print('image_dir is not defined.')
        return

    with open(image_dict_file_path, 'r') as infile:
        image_dict = json.load(infile)

    root = load_xtree_json(xtree_file_path)
    paths = find_all_paths(root=root)
    # Ignore root
    labels = ['_'.join(path) for path in paths[1:]]
    image_dir_set = set(os.listdir(image_dir))
    label_dict = {}
    temp_image_dict = {}
    increment_value = 0
    for label in labels:
        label_dict[label] = increment_value
        increment_value+=1
    
    X = []
    y = []
    counter = 0
    for file_path, value in image_dict.items():
        decoded_file_path = unquote(file_path)  # Decode file path
        basename = os.path.basename(decoded_file_path)
        basename = basename.replace(' ','_')
        
        if basename in image_dir_set:
            
            temp_image_dict[basename] = value
            counter+=1
            continue
        elif os.path.splitext(basename)[0] in [os.path.splitext(filename)[0] for filename in image_dir_set]:
            # Find the file path with the same basename (including extension) in the image_dir_set
            for filename in image_dir_set:
                if os.path.splitext(filename)[0] == os.path.splitext(basename)[0]:
                    
                    temp_image_dict[os.path.join(image_dir, filename)] = value
                    counter+=1
                    break
            continue
        print(basename, 'not found.', counter)
        
    for file_path,labels in temp_image_dict.items():
        X.append(file_path)
        one_hot_labels = [0]*len(label_dict.keys())
        for label in labels:
            label_paths = get_id_paths(root,label)
            label_paths_keys = ['_'.join(label_path) for label_path in label_paths]
            
            for label_path_key in label_paths_keys:
                id = label_dict[label_path_key]
                one_hot_labels[id] = 1
        y.append(one_hot_labels)
    
   
    print(len(X),len(y))
    X = np.array(X)
    y = np.array(y)
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    X_train_whole = []
    y_train_whole = []
    X_test_whole = []
    y_test_whole = []
    for train_index, test_index in msss.split(X, y):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        X_train_whole.extend(list(X_train))
        X_test_whole.extend(list(X_test))
        y_train, y_test = y[train_index], y[test_index]
        y_train_whole.extend(list(y_train))
        y_test_whole.extend(list(y_test))
    
    y_train_labels = []
    y_test_labels = []
    for y_train_one_hot in y_train_whole:
        indices_of_ones = np.where(np.array(y_train_one_hot) == 1)[0]
        temp_labels = []
        for index in indices_of_ones:
            for key,value in label_dict.items():
                if value == index:
                    temp_labels.append(key)
        y_train_labels.append(temp_labels)
    
    for y_test_one_hot in y_test_whole:
        indices_of_ones = np.where(np.array(y_test_one_hot) == 1)[0]
        temp_labels = []
        for index in indices_of_ones:
            for key,value in label_dict.items():
                if value == index:
                    temp_labels.append(key)
        y_test_labels.append(temp_labels)
    
    print(len(X_train_whole))
    print(len(X_test_whole))
    print(len(y_train_labels))
    print(len(y_test_labels))
    print(X_train_whole[0:10])
    print(y_train_labels[0:10])
    train_data = {os.path.basename(file_path): label for file_path, label in zip(X_train_whole, y_train_labels)}
    val_data = {os.path.basename(file_path): label for file_path, label in zip(X_test_whole, y_test_labels)}

    with open(os.path.join(output_path, os.path.splitext(os.path.basename(image_dict_file_path))[0] + '_train.json'), 'w') as outfile:
        json.dump(train_data, outfile)
    with open(os.path.join(output_path, os.path.splitext(os.path.basename(image_dict_file_path))[0] + '_test.json'), 'w') as outfile:
        json.dump(val_data, outfile)
    
    
        
generate_train_and_validation_dataset(image_dict_file_path='../data/image_harnn/fahrzeug/Fahrzeug_image_list.json',xtree_file_path='../data/image_harnn/fahrzeug/Fahrzeug_tree_final.json', output_path='../data/image_harnn/fahrzeug/',image_dir='/home/users/zissoldt/workspace/downloaded_images/')
