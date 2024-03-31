import xtree_utils as xtree
import param_parser as parser
import json, os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
class DatasetAnalyzer():
    def __init__(self, annotation_file_path, hierarchy_file_path, hierarchy_depth,image_count_threshold,path_to_results,dataset_name):
        with open(annotation_file_path,'r') as infile:
            self.image_dict = json.load(infile)
        self.path_to_results = path_to_results
        self.image_count_threshold = image_count_threshold
        self.dataset_name = dataset_name
        
        self.hierarchy = xtree.load_xtree_json(hierarchy_file_path)
        self.hierarchy_dicts = xtree.generate_dicts_per_level(self.hierarchy)
        if hierarchy_depth == -1:
            self.hierarchy_depth = len(self.hierarchy_dicts)
        else:
            self.hierarchy_depth = hierarchy_depth
        self.layer_distribution_dict = []
        self.global_hierarchy_dict = {}
        self.global_distribution_dict = {}
        self.initialize_distribution_dicts(self.hierarchy_dicts)
        self.filtered_hierarchy_dicts = self.filter_hierarchy_dicts_with_threshold()
        self.layer_distribution_dict = []
        self.global_hierarchy_dict = {}
        self.global_distribution_dict = {}
        self.initialize_distribution_dicts(self.filtered_hierarchy_dicts)
        if hierarchy_depth == -1:
            self.hierarchy_depth = len(self.filtered_hierarchy_dicts)
        else:
            self.hierarchy_depth = hierarchy_depth
        for file_name in self.image_dict.keys():
            labels = self.image_dict[file_name]        
            label_dict = self._find_labels_in_hierarchy_dicts(labels,self.filtered_hierarchy_dicts)
            level = 0
            for layer_key in label_dict.keys():
                for label_idx in label_dict[layer_key]:
                    self.layer_distribution_dict[level][label_idx] += 1
                level+=1
            total_class_idxs = self._calc_total_class_labels(label_dict,self.filtered_hierarchy_dicts)
            for total_class_idx in total_class_idxs:
                self.global_distribution_dict[total_class_idx] +=1
                
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
    def generate_global_distribution_plot(self):
        class_distribution_dict = {}
        level = 0
        
        # Choose a colormap
        cmap = cm.get_cmap('tab10')  # You can choose any other colormap from Matplotlib
        
        for layer_dict in self.filtered_hierarchy_dicts:
            layer_distribution_dict = self.layer_distribution_dict[level]
            
            # Extract class names and counts
            classes = [label[label.rfind('_')+1:] for label in layer_dict]
            counts = [layer_distribution_dict[label] for label in layer_dict]
            
            for label, count in zip(classes, counts):
                class_distribution_dict[label] = (count, level)  # Store class count and hierarchy level
            
            level += 1
        
        # Extract class names, counts, and hierarchy levels
        classes = list(class_distribution_dict.keys())
        counts = [class_distribution_dict[label][0] for label in classes]
        hierarchy_levels = [class_distribution_dict[label][1] for label in classes]
        
        # Generate colors dynamically based on the number of hierarchy levels
        colors = [cmap(i) for i in range(len(self.filtered_hierarchy_dicts))]
        
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
        for label in labels:
            path = xtree.get_id_path(self.hierarchy,label)
            
            label_dict = {}
            labels_index = []
            level = 0
            for dict in hierarchy_dicts:
                labels_index = []
                label_dict['layer-{0}'.format(level)] = []
                level +=1
            level = 0
            for i in range(1,self.hierarchy_depth+1):
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
    dataset_analyzer = DatasetAnalyzer(annotation_file_path=args.train_file,hierarchy_file_path=args.hierarchy_file,hierarchy_depth=args.hierarchy_depth,image_count_threshold=args.image_count_threshold,path_to_results=args.path_to_results,dataset_name=args.dataset_name)
    #dataset_analyzer.eval_layer_distribution()
    dataset_analyzer.generate_distribution_per_layer_plot()
    dataset_analyzer.generate_global_distribution_plot()