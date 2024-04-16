import xtree_utils as xtree
import param_parser as parser
import json, os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
import numpy as np
class DatasetAnalyzer():
    def __init__(self, annotation_file_path, hierarchy_depth,image_count_threshold,path_to_results,dataset_name, hierarchy_file_path=None,hierarchy_dicts_file_path=None):
        with open(annotation_file_path,'r') as infile:
            self.image_dict = json.load(infile)
        self.path_to_results = path_to_results
        self.image_count_threshold = image_count_threshold
        self.dataset_name = dataset_name
        self.filtered_hierarchy_dicts = None
        self.hierarchy_dicts = None
        if hierarchy_dicts_file_path is None:
            if hierarchy_file_path is None:
                print('Hierarchy file path and HierarchyDicts file path is None. Not valid!')
                return
            self.load_hierarchy_dicts(hierarchy_file_path=hierarchy_file_path, hierarchy_depth=hierarchy_depth)
            hierarchy_dicts_path = os.path.join(self.path_to_results,self.dataset_name+'_'+str(self.image_count_threshold))
            os.makedirs(hierarchy_dicts_path, exist_ok=True)
            with open(os.path.join(hierarchy_dicts_path,'filtered_hierarchy_dicts.json'), 'w') as outfile:
                json.dump(self.filtered_hierarchy_dicts, outfile)
        else:
            self.load_hierarchy_dicts_from_file(hierarchy_dicts_file_path=hierarchy_dicts_file_path,hierarchy_file_path=hierarchy_file_path,hierarchy_depth=hierarchy_depth)
        self.num_classes_list = [len(list(hierarchy_dict)) for hierarchy_dict in self.filtered_hierarchy_dicts]
        self.total_class_num = sum(self.num_classes_list)
        print('Dataset Size:',sum([image_count for image_count in self.layer_distribution_dict[0].values()]))
        print('Num Classes List:',self.num_classes_list)
        print('Total Class Num',self.total_class_num)
    def load_hierarchy_dicts(self,hierarchy_file_path,hierarchy_depth):
        self.hierarchy = xtree.load_xtree_json(hierarchy_file_path)
        self.hierarchy_dicts = xtree.generate_dicts_per_level(self.hierarchy)
        if hierarchy_depth == -1:
            self.hierarchy_depth = len(self.hierarchy_dicts)
        else:
            self.hierarchy_depth = hierarchy_depth
        
        
        self.initialize_distribution_dicts(self.hierarchy_dicts)
        self.filtered_hierarchy_dicts = self.filter_hierarchy_dicts_with_threshold()
        
        self.initialize_distribution_dicts(self.filtered_hierarchy_dicts)
        if hierarchy_depth == -1:
            self.hierarchy_depth = len(self.filtered_hierarchy_dicts)
        else:
            self.hierarchy_depth = hierarchy_depth
        self.eval_distribution_dicts()
    def load_hierarchy_dicts_from_file(self,hierarchy_dicts_file_path,hierarchy_file_path,hierarchy_depth):
        self.hierarchy = xtree.load_xtree_json(hierarchy_file_path)
        with open(hierarchy_dicts_file_path,'r') as infile:
            self.filtered_hierarchy_dicts = json.load(infile)
        if hierarchy_depth == -1:
            self.hierarchy_depth = len(self.filtered_hierarchy_dicts)
        else:
            self.hierarchy_depth = hierarchy_depth
        self.initialize_distribution_dicts(self.filtered_hierarchy_dicts)
        self.eval_distribution_dicts()
    def initialize_distribution_dicts(self,hierarchy_dicts):
        self.layer_distribution_dict_explicit= []
        self.layer_distribution_dict = []
        self.global_hierarchy_dict = {}
        self.global_distribution_dict = {}
        for hierarchy_dict in hierarchy_dicts:
            layer_dict = {}
            explicit_layer_dict ={}
            for key in hierarchy_dict:
                layer_dict[hierarchy_dict[key]] = 0
                explicit_layer_dict[hierarchy_dict[key]] = 0
            self.layer_distribution_dict.append(layer_dict)
            self.layer_distribution_dict_explicit.append(explicit_layer_dict)
            
        counter = 0
        for hierarchy_dict in hierarchy_dicts:
            for key in hierarchy_dict:
                self.global_hierarchy_dict[key] = counter
                self.global_distribution_dict[counter] = 0
                counter+=1
    
    def eval_distribution_dicts(self):
        for file_name in self.image_dict.keys():
            labels = self.image_dict[file_name]        
            label_dict = self._find_labels_in_hierarchy_dicts(labels,self.filtered_hierarchy_dicts)
            self._eval_labels_in_hierarchy_dicts(labels=labels,hierarchy_dicts=self.filtered_hierarchy_dicts)
            level = 0
            for layer_key in label_dict.keys():
                for label_idx in label_dict[layer_key]:
                    self.layer_distribution_dict[level][label_idx] += 1
                level+=1
            total_class_idxs = self._calc_total_class_labels(label_dict,self.filtered_hierarchy_dicts)
            for total_class_idx in total_class_idxs:
                self.global_distribution_dict[total_class_idx] +=1
    
    def filter_hierarchy_dicts_with_threshold(self):
        counter = 0        
        for file_name in self.image_dict.keys():
            labels = self.image_dict[file_name]        
            label_dict = self._find_labels_in_hierarchy_dicts(labels,self.hierarchy_dicts)
            level = 0
            
            for layer_key in label_dict.keys():
                for label_idx in label_dict[layer_key]:
                    if level == 4 and label_idx == 1:
                        counter+=1
                    self.layer_distribution_dict[level][label_idx] += 1
                level+=1
            total_class_idxs = self._calc_total_class_labels(label_dict,self.hierarchy_dicts)
            for total_class_idx in total_class_idxs:
                self.global_distribution_dict[total_class_idx] +=1
        print('Counter:',counter)
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
    def generate_distribution_per_layer_plot(self):
        level=0
        for layer_dict in self.filtered_hierarchy_dicts:
            layer_distribution_dict = self.layer_distribution_dict[level]
            class_distriubtion_dict = {}
            for label in layer_dict:
                class_distriubtion_dict[label] = layer_distribution_dict[layer_dict[label]]

            # Sort classes based on their counts
            sorted_classes = sorted(class_distriubtion_dict.items(), key=lambda x: x[1], reverse=True)
            classes = [x[0][x[0].rfind('_')+1:] for x in sorted_classes]
            counts = [x[1] for x in sorted_classes]
            print(counts)
            #classes = [label[label.rfind('_')+1:] for label in list(class_distriubtion_dict.keys())]
            #data_points = list(class_distriubtion_dict.values())
            scaling_factor = min(10,max(2, 100 / len(classes)))
            bars = plt.bar(classes, counts, color='skyblue')
            plt.xlabel('Klassen')
            plt.ylabel('Anzahl')
            plt.title('Verteilung der Klassen für Schicht {0}'.format(level+1))
            plt.xticks(rotation=90)  # Rotate class names for better readability if needed
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()

            # Set font size for class labels
            for tick in plt.gca().xaxis.get_major_ticks():
                tick.label.set_fontsize(scaling_factor)
            fig_path = os.path.join(self.path_to_results,self.dataset_name+'_'+str(self.image_count_threshold))
            fig_file_name='distribution_layer_{0}_plot.png'.format(level+1)
            fig_file_path = os.path.join(fig_path,fig_file_name)
            os.makedirs(fig_path, exist_ok=True)
            plt.savefig(fig_file_path)
            plt.clf()
            level+=1
            
    def generate_explicit_distribution_plot_per_layer(self):
        class_distriubtion_dict = [{} for filtered_hierarchy_dict in self.filtered_hierarchy_dicts]
        level = 0

        # Determine the number of hierarchy levels
        num_levels = len(self.filtered_hierarchy_dicts)

        # Choose a colormap
        cmap = cm.get_cmap('tab10')  # You can choose any other colormap from Matplotlib

        for layer_dict in self.filtered_hierarchy_dicts:
            layer_distribution_dict = self.layer_distribution_dict_explicit[level]

            for label in layer_dict:
                class_distriubtion_dict[level][label] = layer_distribution_dict[layer_dict[label]]
            level += 1
        classes = []
        
        #print(self.layer_distribution_dict)
        for i in range(len(self.layer_distribution_dict)):
            temp_str = f'{i+1}:'
            for key,value in self.layer_distribution_dict[i].items():
                temp_str+=str(value)+','
            print(temp_str)
        #print(self.layer_distribution_dict_explicit)
        for level in range(len(self.filtered_hierarchy_dicts)):
            classes=tuple([x[x.rfind('_')+1:] for x in self.filtered_hierarchy_dicts[level].keys()])
            
            weight_counts = {}
            for layer in range(level,len(self.filtered_hierarchy_dicts)):
                weight_counts[f'Hierarchy-Layer-{layer+1}'] = []
                
            for key in self.filtered_hierarchy_dicts[level].keys():
                
                for i in range(level, len(self.filtered_hierarchy_dicts)):
                    summed_image_count = 0
                    for label in self.filtered_hierarchy_dicts[i].keys():
                        if label.startswith(key):
                            summed_image_count+=class_distriubtion_dict[i][label]
                        
                    weight_counts[f'Hierarchy-Layer-{i+1}'].append(summed_image_count)
            # Transpose the data
            transposed_data = list(zip(*weight_counts.values()))
            # Calculate the sum of each column
            column_sums = [sum(column) for column in transposed_data]
            # Sort columns by their sums in descending order
            sorted_columns = sorted(enumerate(column_sums), key=lambda x: x[1], reverse=True)
            # Permutate data values based on sorted column indices
            permutated_data = {key: [weight_counts[key][index] for index, _ in sorted_columns] for key in weight_counts.keys()}
            for key,value in permutated_data.items():
                print(f'{key}:{value}')
            print('Verteilung der Klassen für Schicht {0}'.format(level+1))
            
            permutated_classes = [classes[index] for index, _ in sorted_columns]
            weight_counts = permutated_data
            width = 0.5

            fig, ax = plt.subplots()
            bottom = np.zeros(len(classes))
            scaling_factor = min(10,max(2, 100 / len(permutated_classes)))
            for boolean, weight_count in weight_counts.items():
                p = ax.bar(permutated_classes, weight_count, width, label=boolean, bottom=bottom)
                bottom += weight_count
            plt.xlabel('Klassen')
            plt.ylabel('Anzahl')
            plt.title('Verteilung der Klassen für Schicht {0}'.format(level+1))
            plt.xticks(rotation=90)  # Rotate class names for better readability if needed
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.legend(loc="upper right")
            # Set font size for class labels
            for tick in plt.gca().xaxis.get_major_ticks():
                tick.label.set_fontsize(scaling_factor)
            fig_path = os.path.join(self.path_to_results,self.dataset_name+'_'+str(self.image_count_threshold))
            fig_file_name='explicit_distribution_layer_{0}_plot.png'.format(level+1)
            fig_file_path = os.path.join(fig_path,fig_file_name)
            os.makedirs(fig_path, exist_ok=True)
            plt.savefig(fig_file_path)
            plt.clf()
    def generate_global_distribution_plot(self):
        class_distriubtion_dict = {}
        level = 0

        # Determine the number of hierarchy levels
        num_levels = len(self.filtered_hierarchy_dicts)

        # Choose a colormap
        cmap = cm.get_cmap('tab10')  # You can choose any other colormap from Matplotlib

        for layer_dict in self.filtered_hierarchy_dicts:
            layer_distribution_dict = self.layer_distribution_dict[level]

            for label in layer_dict:
                class_distriubtion_dict[label] = (layer_distribution_dict[layer_dict[label]], level)  # Store class count and hierarchy level
            level += 1

        # Sort classes based on their counts
        sorted_classes = [item for item in class_distriubtion_dict.items()]
        
        # Extract class names, counts, and hierarchy levels
        classes = [x[0][x[0].rfind('_')+1:] for x in sorted_classes]
        counts = [x[1][0] for x in sorted_classes]
        hierarchy_levels = [x[1][1] for x in sorted_classes]

        # Generate colors dynamically based on the number of hierarchy levels
        colors = [cmap(i) for i in range(num_levels)]

        # Create bars with different colors for each hierarchy level
        fig, ax = plt.subplots()
        bars = ax.bar(classes, counts, color=[colors[level] for level in hierarchy_levels])

        plt.xlabel('Klassen')
        plt.ylabel('Anzahl')
        plt.title('Globale Verteilung der Klassen')
        plt.xticks([])  # Rotate class names for better readability if needed
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Save the plot
        fig_path = os.path.join(self.path_to_results, self.dataset_name + '_' + str(self.image_count_threshold))
        fig_file_name = 'global_distribution_plot.png'
        fig_file_path = os.path.join(fig_path, fig_file_name)
        os.makedirs(fig_path, exist_ok=True)
        plt.savefig(fig_file_path)
        plt.clf()
    def _find_labels_in_hierarchy_dicts(self,labels,hierarchy_dicts):
        label_dict = {}
        level = 0
        for dict in hierarchy_dicts:
            labels_index = []
            label_dict['layer-{0}'.format(level)] = []
            level +=1
            
        for label in labels:
            path = xtree.get_id_path(self.hierarchy,label)
            
            
            labels_index = []
            level = 0
            
            for i in range(1,self.hierarchy_depth+1):
                temp_key = '_'.join(path[:i+1])
                temp_dict = hierarchy_dicts[i-1]
                if temp_key not in temp_dict.keys():
                    break
                temp_label = temp_dict[temp_key]
                label_dict['layer-{0}'.format(i-1)].append(temp_label)

                            
                         
            #label_dict['layer-{0}'.format(level)].extend(labels_index)
            #level+=1
        
        for level_key in label_dict.keys():
            label_dict[level_key] = list(set(label_dict[level_key]))
        
        return label_dict

    def _eval_labels_in_hierarchy_dicts(self,labels,hierarchy_dicts):
        def is_subset(path1, path2):
        # Check if path1 is a subset of path2
            return set(path1).issubset(set(path2))
        paths = []
        for label in labels:          
            path = xtree.get_id_path(self.hierarchy,label)[:len(hierarchy_dicts)+1]
            paths.append(path)
        redundant_paths = []

        for i, path1 in enumerate(paths):
            for j, path2 in enumerate(paths):
                if i != j:
                    if set(path1) == set(path2):
                       break
                    if is_subset(path1, path2):
                        redundant_paths.append(path1)
                        break  # Break after finding the first instance of a subset

        # Remove redundant paths
        if len(redundant_paths) != 0:
            unique_paths = [path for path in paths if path not in redundant_paths]
            unique_paths.sort()
            unique_paths=list(unique_paths for unique_paths,_ in itertools.groupby(unique_paths))
        
        else:
            unique_paths = paths
        for unique_path in unique_paths:
            level = len(unique_path)-1
            hierarchy_dict = hierarchy_dicts[level-1]
            hierarchy_key = '_'.join(unique_path)
            if hierarchy_key in hierarchy_dict:
                label_idx = hierarchy_dict[hierarchy_key]
                self.layer_distribution_dict_explicit[level-1][label_idx] +=1
                if level == 6 and label_idx == 1:
                    print(self.layer_distribution_dict_explicit[level-1][label_idx])
                    print(unique_path)
                    print(unique_paths)
        
        
        
    
    def _get_last_label_in_hierarchy_dicts(self,hierarchy_dicts):
        level = len(list(hierarchy_dicts.keys()))
        for key in reversed(list(hierarchy_dicts.keys())):
            if len(hierarchy_dicts[key]) > 0:
                return 
                 
            
    
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
def plot_distribution(label_dict,distribution_dict):
        class_distriubtion_dict = {}
        for label in label_dict:
            class_distriubtion_dict[label] = distribution_dict[label_dict[label]]
        
        classes = [label[label.rfind('_')+1:] for label in list(class_distriubtion_dict.keys())]
        data_points = list(class_distriubtion_dict.values())
        plt.bar(classes, data_points, color='skyblue')
        plt.xlabel('Classes')
        plt.ylabel('Counts')
        plt.title('Distribution of Classes')
        plt.xticks(rotation=90)  # Rotate class names for better readability if needed
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        #plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    args = parser.dataset_analyzer_parser()    
    dataset_analyzer = DatasetAnalyzer(annotation_file_path=args.train_file,hierarchy_file_path=args.hierarchy_file,hierarchy_dicts_file_path=args.hierarchy_dicts_file,hierarchy_depth=args.hierarchy_depth,image_count_threshold=args.image_count_threshold,path_to_results=args.path_to_results,dataset_name=args.dataset_name)
    dataset_analyzer.generate_explicit_distribution_plot_per_layer()
    dataset_analyzer.generate_distribution_per_layer_plot()
    dataset_analyzer.generate_global_distribution_plot()