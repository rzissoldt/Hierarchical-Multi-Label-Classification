import xtree_utils as xtree
import param_parser as parser
import json, os
import matplotlib.pyplot as plt

class DatasetAnalyzer():
    def __init__(self, annotation_file_path, hierarchy_file_path, hierarchy_depth,image_count_threshold,path_to_results,dataset_name):
        with open(annotation_file_path,'r') as infile:
            self.image_dict = json.load(infile)
        self.path_to_results = path_to_results
        self.image_count_threshold = image_count_threshold
        self.dataset_name = dataset_name
        self.hierarchy_depth = hierarchy_depth
        self.hierarchy = xtree.load_xtree_json(hierarchy_file_path)
        self.hierarchy_dicts = xtree.generate_dicts_per_level(self.hierarchy)
        self.filtered_hierarchy_dicts = xtree.filter_hierarchy_dict_with_threshold(self.hierarchy_dicts,image_count_threshold=image_count_threshold,hierarchy_depth=hierarchy_depth)
        self.layer_distribution_dict = []
        self.global_hierarchy_dict = {}
        self.global_distribution_dict = {}
    def eval_layer_distribution(self):
        for hierarchy_dict in self.filtered_hierarchy_dicts:
            layer_dict = {}
            for key in hierarchy_dict:
                layer_dict[hierarchy_dict[key]] = 0
            self.layer_distribution_dict.append(layer_dict)
            
        counter = 0
        for hierarchy_dict in self.filtered_hierarchy_dicts:
            for key in hierarchy_dict:
                self.global_hierarchy_dict[key] = counter
                self.global_distribution_dict[counter] = 0
                counter+=1
            
        
        for file_name in self.image_dict.keys():
            labels = self.image_dict[file_name]        
            label_dict = self._find_labels_in_hierarchy_dicts(labels)
            level = 0
            for layer_key in label_dict.keys():
                for label_idx in label_dict[layer_key]:
                    self.layer_distribution_dict[level][label_idx] += 1
                level+=1
            total_class_idxs = self._calc_total_class_labels(label_dict)
            for total_class_idx in total_class_idxs:
                self.global_distribution_dict[total_class_idx] +=1
        #plot_distribution(self.global_hierarchy_dict,self.global_distribution_dict)
        #plot_distribution(self.filtered_hierarchy_dicts[1],self.layer_distribution_dict[1])
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
        class_distriubtion_dict = {}
        for label in self.global_hierarchy_dict:
            class_distriubtion_dict[label] = self.global_distribution_dict[self.global_hierarchy_dict[label]]
        # Sort classes based on their counts
        sorted_classes = sorted(class_distriubtion_dict.items(), key=lambda x: x[1], reverse=True)
        classes = [x[0][x[0].rfind('_')+1:] for x in sorted_classes]
        counts = [x[1] for x in sorted_classes]
        bars = plt.bar(classes, counts, color='skyblue')
        plt.xlabel('Klassen')
        plt.ylabel('Anzahl')
        plt.title('Globale Verteilung der Klassen')
        plt.xticks([])  # Rotate class names for better readability if needed
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        fig_path = os.path.join(self.path_to_results,self.dataset_name+'_'+str(self.image_count_threshold))
        fig_file_name='global_distribution_plot.png'
        fig_file_path = os.path.join(fig_path,fig_file_name)
        os.makedirs(fig_path, exist_ok=True)
        plt.savefig(fig_file_path)
        plt.clf()
    def _find_labels_in_hierarchy_dicts(self,labels):
        for label in labels:
            path = xtree.get_id_path(self.hierarchy,label)
            
            label_dict = {}
            labels_index = []
            level = 0
            for dict in self.filtered_hierarchy_dicts:
                labels_index = []
                label_dict['layer-{0}'.format(level)] = []
                level +=1
            level = 0
            for i in range(1,self.hierarchy_depth+1):
                temp_key = '_'.join(path[:i+1])
                temp_dict = self.filtered_hierarchy_dicts[i-1]
                if temp_key not in temp_dict.keys():
                    break
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
                        total_class_label+=len(self.filtered_hierarchy_dicts[i].keys())
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
    dataset_analyzer.eval_layer_distribution()
    dataset_analyzer.generate_distribution_per_layer_plot()
    dataset_analyzer.generate_global_distribution_plot()