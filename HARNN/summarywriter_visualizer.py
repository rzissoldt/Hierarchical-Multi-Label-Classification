from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import random
from torchvision import transforms
import os, json
import torch
from types import SimpleNamespace
# Add the parent directory to the Python path
sys.path.append('../')
from HARNN.model.baseline_model import BaselineModel
from HARNN.model.chmcnn_model import ConstrainedFFNNModel
from HARNN.model.hmcnet_model import HmcNet
from HARNN.model.buhcapsnet_model import BUHCapsNet
from HARNN.dataset.hierarchy_dataset import HierarchyDataset
from torchmetrics.classification import Precision, Recall, F1Score
from utils.data_helpers import precision_recall_f1_score
from utils import xtree_utils as xtree
from utils import data_helpers as dh
from utils import param_parser as parser
from PIL import Image
def extract_file_with_prefix(directory, prefix):
    # Get the list of files in the directory
    files = os.listdir(directory)
    
    # Filter files that start with the specified prefix
    matching_files = [file for file in files if file.startswith(prefix)]
    
    return matching_files
def get_event_file_from_dir(model_dir):
    event_file_prefix = 'events.out.tfevents'
    model_config_file = 'model_config.json'
    matching_event_file = extract_file_with_prefix(model_dir,event_file_prefix)[0]
    event_file_path = os.path.join(model_dir, matching_event_file)
    return event_file_path

def get_level_metrics_from_event_file(event_file_path,hierarchy_depth):
    loader = EventFileLoader(event_file_path)
    macro_precision_list = []
    macro_recall_list = []
    macro_f1_list = []
    macro_auroc_list = []
    macro_auprc_list = []
    micro_precision_list = []
    micro_recall_list = []
    micro_f1_list = []
    micro_auroc_list = []
    micro_auprc_list = []
    
    for event in loader.Load():
        summary_value = event.summary.value
        if len(summary_value) == 0:
            continue
        
        summary_value = summary_value[0]
        # Check if the attribute 'summary' exists
        if hasattr(summary_value, 'tag'):
            for i in range(hierarchy_depth):
                if summary_value.tag == f'Validation/{i+1}-LayerMacroPrecision':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        macro_precision_list.append(float_val)
                if summary_value.tag == f'Validation/{i+1}-LayerMacroRecall':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        macro_recall_list.append(float_val)
                if summary_value.tag == f'Validation/{i+1}-LayerMacroF1':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        macro_f1_list.append(float_val)
                if summary_value.tag == f'Validation/{i+1}-LayerMacroAUC':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        macro_auroc_list.append(float_val)
                if summary_value.tag == f'Validation/{i+1}-LayerMacroAUPRC':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        macro_auprc_list.append(float_val)
                if summary_value.tag == f'Validation/{i+1}-LayerMicroPrecision':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        micro_precision_list.append(float_val)
                if summary_value.tag == f'Validation/{i+1}-LayerMicroRecall':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        micro_recall_list.append(float_val)
                if summary_value.tag == f'Validation/{i+1}-LayerMicroF1':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        micro_f1_list.append(float_val)
                if summary_value.tag == f'Validation/{i+1}-LayerMicroAUC':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        micro_auroc_list.append(float_val)
                if summary_value.tag == f'Validation/{i+1}-LayerMicroAUPRC':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        micro_auprc_list.append(float_val)
    metrics = {
        'micro_precision':micro_precision_list,
        'micro_recall':micro_recall_list,
        'micro_f1':micro_f1_list,
        'micro_auroc':micro_auroc_list,
        'micro_auprc':micro_auprc_list,
        'macro_precision':macro_precision_list,
        'macro_recall':macro_recall_list,
        'macro_f1':macro_f1_list,
        'macro_auroc':macro_auroc_list,
        'macro_auprc':macro_auprc_list
    }         
    return metrics

"""
def save_hierarchy_level_metric_plot(hierarchy_level_metrics, metric_key, level, output_path):
    temp_metric_list = []
    for model_name in hierarchy_level_metrics.keys():
        metric_dict = hierarchy_level_metrics[model_name]
        temp_metric_list.append(metric_dict[metric_key][level])
            
    data = {
        'Model': list(hierarchy_level_metrics.keys()),
        f'Hierarchy_Level_{level+1}': temp_metric_list,
    }
    
    # Convert data to pandas DataFrame
    df = pd.DataFrame(data)
    
    # Set 'Model' column as index
    df.set_index('Model', inplace=True)
    
    # Define color for each bar using a dynamic color gradient
    num_models = len(df)
    cmap = plt.cm.viridis
    colors = ['red','yellow','blue']
    
    # Plotting
    ax = df.plot(kind='bar', width=0.9, color=colors, legend=False)
    plt.title('Metrics per Hierarchy Level')
    plt.xlabel('Models')
    plt.ylabel(f'{metric_key}')
    plt.xticks(rotation=45)
    
    # Create directory for the plot if it doesn't exist
    plot_dir = os.path.join(output_path, f'level{level+1}')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Save the plot
    plot_file_path = os.path.join(plot_dir, f'{metric_key}_level{level+1}.png')
    plt.tight_layout()  # Adjust layout for better visualization
    plt.savefig(plot_file_path)
    plt.close()
    """
def save_hierarchy_level_metric_plot(hierarchy_level_metrics, metric_key, level, output_path):
    temp_metric_list = []
    model_names = list(hierarchy_level_metrics.keys())
    num_models = len(model_names)
    for model_name in model_names:
        metric_dict = hierarchy_level_metrics[model_name]
        temp_metric_list.append(metric_dict[metric_key][level])

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.5  # Adjusted width to make bars half the width
    index = np.arange(num_models)  # Positions of the bars
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, num_models))
    
    ax.bar(index, temp_metric_list, bar_width, color=colors, align='edge')
    
    # Customize plot
    ax.set_title('Metrics per Hierarchy Level')
    ax.set_xlabel('Models')
    ax.set_ylabel(metric_key)
    ax.set_xticks(index + bar_width / 2)  # Positions of the model names
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)  # Labels under the center of bars

    # Create directory for the plot if it doesn't exist
    plot_dir = os.path.join(output_path, f'level{level+1}')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Save the plot
    plot_file_path = os.path.join(plot_dir, f'{metric_key}_level{level+1}.png')
    plt.tight_layout()  # Adjust layout for better visualization
    plt.savefig(plot_file_path)
    plt.close()

def save_hierarchy_metric_plot(hierarchy_level_metrics, metric_key, hierarchy_depth, plot_name, output_path):
    # Sample data (replace with your actual data)
    data = {
        'Hierarchy_Level': [f'Level-{i+1}' for i in range(hierarchy_depth)],
    }
    
    for model_name in hierarchy_level_metrics.keys():
        temp_metric_list = []
        for i in range(hierarchy_depth): 
            metric_dict = hierarchy_level_metrics[model_name]
            temp_metric_list.append(metric_dict[metric_key][i])
        data[f'{model_name}'] = temp_metric_list
    
    # Convert data to pandas DataFrame
    df = pd.DataFrame(data)

    # Set 'Model' column as index
    df.set_index('Hierarchy_Level', inplace=True)

    # Plotting
    ax = df.plot(kind='bar')
    plt.title(f'{metric_key} für {plot_name}', fontsize=10)
    plt.xlabel('Hierarchy Level')
    plt.ylabel(metric_key)
    plt.xticks(rotation=45)
    
    # Move legend outside the plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Create directory for the plot if it doesn't exist
    plot_dir = os.path.join(output_path)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Save the plot
    plot_file_path = os.path.join(plot_dir, f'{metric_key}.png')
    
    # Save the figure
    plt.savefig(plot_file_path, bbox_inches='tight')

    # Show the plot
    plt.close()


def visualize_sample_image(image_file_path,true_label,model_names,best_model_dirs,threshold,hierarchy_dicts,output_file_path,explicit_hierarchy,num_classes_list,device):
    
    #os.makedirs(output_file_path, exist_ok=True)
    transform = transforms.Compose([
            transforms.Resize((256, 256)),                    
            transforms.CenterCrop(224),                       
            transforms.ToTensor(),                             
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                             std=[0.229, 0.224, 0.225])
        ])
    image = Image.open(image_file_path)

    # Convert to RGB if it isn't already
    if image.mode != 'RGB':
        image = image.convert('RGB')
      # Initialize pil_image with the original image
    transformed_image = transform(image).to(device)
    total_class_num = true_label.shape[0]
    batch_tensor = transformed_image.unsqueeze(0)
    counter = 0
    score_list = []
    hmcnet_recall = 0
    chmcnn_recall = 0
    micro_recall = Recall(task="multilabel", average='micro', num_labels=total_class_num, threshold=0.5)
    for model_name in model_names:
        
        best_model_config_path = os.path.join(best_model_dirs[counter],'model_config.json')
        with open(best_model_config_path, 'r') as infile:
            best_model_config = SimpleNamespace(**json.load(infile))
        best_model_file_name = os.listdir(os.path.join(best_model_dirs[counter],'models'))[0]
        best_model_file_path = os.path.join(best_model_dirs[counter],'models',best_model_file_name)
        
        if model_name == 'baseline':
            model = BaselineModel(output_dim=total_class_num, args=best_model_config).to(device=device)    
            # Load Best Model Paramsbest_model_config
            
            best_checkpoint = torch.load(best_model_file_path)
            model.load_state_dict(best_checkpoint)
            model.eval()
            score = model(batch_tensor.float())
            thresholded_score = score > threshold
            score_list.append(thresholded_score)
            
        elif model_name == 'chmcnn':
            model = ConstrainedFFNNModel(output_dim=total_class_num,R=explicit_hierarchy, args=best_model_config).to(device=device) 
            
            # Load Best Model Params
            best_checkpoint = torch.load(best_model_file_path)
            model.load_state_dict(best_checkpoint)
            model.training =False
            model.eval()
            score = model(batch_tensor.float())
            thresholded_score = score > threshold
            score_list.append(thresholded_score)
            thresholded_score = thresholded_score.to('cpu').numpy().astype(int)
            thresholded_score_tensor_chmcnn = torch.tensor(thresholded_score)
            chmcnn_recall = micro_recall(thresholded_score_tensor_chmcnn, torch.tensor(true_label).unsqueeze(0))
            
        elif model_name == 'hmcnet':
            
            model = HmcNet(global_average_pooling_active=best_model_config.is_backbone_global_average_pooling_active,feature_dim=best_model_config.feature_dim_backbone,attention_unit_size=best_model_config.attention_dim,backbone_fc_hidden_size=best_model_config.backbone_dim,fc_hidden_size=best_model_config.fc_dim,freeze_backbone=True,highway_fc_hidden_size=best_model_config.highway_fc_dim,highway_num_layers=best_model_config.highway_num_layers,num_classes_list=num_classes_list,total_classes=total_class_num,l2_reg_lambda=best_model_config.l2_lambda,dropout_keep_prob=best_model_config.dropout_rate,alpha=best_model_config.alpha,beta=best_model_config.beta,device=device,is_backbone_embedding_active=best_model_config.is_backbone_embedding_active).to(device=device)
            # Load Best Model Params
            
            best_checkpoint = torch.load(best_model_file_path)
            model.load_state_dict(best_checkpoint)
            model.eval()
            score, _, _ = model(batch_tensor)
            thresholded_score = score > threshold
            score_list.append(thresholded_score)
            thresholded_score = thresholded_score.to('cpu').numpy().astype(int)
            thresholded_score_tensor_hmcnet = torch.tensor(thresholded_score)
            hmcnet_recall = micro_recall(thresholded_score_tensor_hmcnet, torch.tensor(true_label).unsqueeze(0))
        elif model_name == 'buhcapsnet':
            model = BUHCapsNet(pcap_n_dims=best_model_config.pcap_n_dims,scap_n_dims=best_model_config.scap_n_dims,num_classes_list=num_classes_list,routings=best_model_config.routing_iterations,args=best_model_config,device=device).to(device=device)    
            best_checkpoint = torch.load(best_model_file_path)
            model.load_state_dict(best_checkpoint)
            model.eval()
            score = model(batch_tensor)
            score = torch.cat(score,dim=0).unsqueeze(0)
            thresholded_score = score > threshold
            score_list.append(thresholded_score)
        counter +=1

    if hmcnet_recall - chmcnn_recall > 1e-16:
        print('Recall is higher')
        return
    
    
    # Festlegen der Größe des Ausgabebildes
    plt.figure(figsize=(8, 8))  # Breite: 8 Zoll, Höhe: 6 Zoll
    # Anzeigen des Bildes
    # Resize the image
    image = image.resize((500, 500), Image.Resampling.LANCZOS)
    plt.imshow(image)
    image_np = np.array(image)
    
    
    plt.axis('off')  # Achsen ausschalten
    swapped_hierarchy_dict = [{v: k for k, v in hierarchy_dict.items()} for hierarchy_dict in hierarchy_dicts]
   
    # Text für die richtigen Labels
    base_text_anchor = image_np.shape[0] + 15
    
    
    for k in range(len(model_names)):
        start_index = 0
        plt.text(0,base_text_anchor,f'{model_names[k]}',fontsize=11,weight='bold')
        base_text_anchor = base_text_anchor + 15
        thresholded_score = score_list[k][0].to('cpu').numpy().astype(int)
        for i in range(len(swapped_hierarchy_dict)):
            plt.text(0,base_text_anchor,f'Hierarchy-Layer-{i+1}:',fontsize=9,weight='bold')
            anchor_counter = 0
            for j in swapped_hierarchy_dict[i].keys():
                wk_id = swapped_hierarchy_dict[i][j].split('_')[-1]
                if true_label[start_index+j] == 1 and thresholded_score[start_index+j] == 1:
                    plt.text(120+(anchor_counter)*60,base_text_anchor,f'{wk_id}',color='green',fontsize=9)
                    anchor_counter+=1
                elif true_label[start_index+j] == 1 and thresholded_score[start_index+j] == 0:
                    plt.text(120+(anchor_counter)*60,base_text_anchor,f'{wk_id}',color='red',fontsize=9)
                    anchor_counter+=1
                elif true_label[start_index+j] == 0 and thresholded_score[start_index+j] == 1:
                    plt.text(120+(anchor_counter)*60,base_text_anchor,f'{wk_id}',color='orange',fontsize=9)
                    anchor_counter+=1
            base_text_anchor+=15
            start_index+=len(swapped_hierarchy_dict[i])
        base_text_anchor += 15
        thresholded_score_tensor = torch.tensor(thresholded_score)
        true_label_tensor = torch.tensor(true_label)
        pre_micro, rec_micro, f1_micro = precision_recall_f1_score(binary_predictions=thresholded_score_tensor,labels=true_label_tensor, average='micro')
        pre_macro, rec_macro, f1_macro = precision_recall_f1_score(binary_predictions=thresholded_score_tensor,labels=true_label_tensor, average='micro')
        plt.text(200,base_text_anchor,f'Precision: {pre_micro}, {pre_macro}',fontsize=9,weight='bold')
        plt.text(350,base_text_anchor+25,f'Recall: {rec_micro}, {rec_macro}',fontsize=9,weight='bold')
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='True Positive'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='False Negative'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='False Positive')
    ]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(output_file_path,bbox_inches='tight')
    plt.clf()
    print('Image saved')
    image.close()
def visualize_sample_images(images,true_labels,scores,threshold,hierarchy_dicts,output_file_path):
    
    #os.makedirs(output_file_path, exist_ok=True)
    for i in range(len(images)):
        score = scores[i]
        thresholded_score = score > threshold
        reverse_transform = transforms.Compose([
            transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                         std=[1/0.229, 1/0.224, 1/0.225]),
         transforms.Normalize(mean=[-0.5, -0.5, -0.5],
                         std=[1, 1, 1]),
         transforms.ToPILImage()
        ])
        image = reverse_transform(images[i])
        # Anzeigen des Bildes
        #image_np = image.cpu().numpy()
        #image_np = image_np.transpose(1,2,0)
        true_label = true_labels[i]
        # Festlegen der Größe des Ausgabebildes
        plt.figure(figsize=(8, 6))  # Breite: 8 Zoll, Höhe: 6 Zoll
        # Anzeigen des Bildes
        plt.imshow(image)
        image_np = np.array(image)
        plt.axis('off')  # Achsen ausschalten
        swapped_hierarchy_dict = [{v: k for k, v in hierarchy_dict.items()} for hierarchy_dict in hierarchy_dicts]
        # Text für die richtigen Labels
        base_text_anchor = image_np.shape[1] + 20
        start_index = 0
        for i in range(len(swapped_hierarchy_dict)):
            plt.text(0,base_text_anchor,f'Hierarchy-Layer-{i+1}:',fontsize=9,weight='bold')
            anchor_counter = 0
            for j in swapped_hierarchy_dict[i].keys():
                wk_id = swapped_hierarchy_dict[i][j].split('_')[-1]
                if true_label[start_index+j] == 1 and true_label[start_index+j] == thresholded_score[start_index+j]:
                    plt.text(35+(anchor_counter+1)*38,base_text_anchor,f'{wk_id}',color='green',fontsize=9)
                    anchor_counter+=1
                elif true_label[start_index+j] == 1 and true_label[start_index+j] != thresholded_score[start_index+j]:
                    plt.text(35+(anchor_counter+1)*38,base_text_anchor,f'{wk_id}',color='red',fontsize=9)
                    anchor_counter+=1
                elif true_label[start_index+j] == 0 and true_label[start_index+j] != thresholded_score[start_index+j]:
                    plt.text(35+(anchor_counter+1)*38,base_text_anchor,f'{wk_id}',color='orange',fontsize=9)
                    anchor_counter+=1
            base_text_anchor+=10
            start_index+=len(swapped_hierarchy_dict[i])

        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='True Positive'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='False Negative'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='False Positive')
        ]
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(os.path.join(output_file_path,f'sample_image{i}'),bbox_inches='tight')
        plt.clf()
        
def visualize_test_results(args):
    if len(args.result_model_dirs) != len(args.model_names):
        print('Model dirs list and model names list must have same size!')
        return
    counter = 0
    models_metric_dict = {}
    for result_model_dir in args.result_model_dirs:
        result_model_file_path = get_event_file_from_dir(result_model_dir)
        models_metric_dict[args.model_names[counter]] = get_level_metrics_from_event_file(result_model_file_path,args.hierarchy_depth)
        counter+=1
    metric_keys = [
        'micro_precision',
        'micro_recall',
        'micro_f1',
        'micro_auroc',
        'micro_auprc',
        'macro_precision',
        'macro_recall',
        'macro_f1',
        'macro_auroc',
        'macro_auprc'
    ]
    os.makedirs(args.output_dir, exist_ok=True)
    for level in range(args.hierarchy_depth):
        for metric_key in metric_keys:
            save_hierarchy_metric_plot(models_metric_dict,metric_key,args.hierarchy_depth,args.plot_name,args.output_dir)


if __name__ == '__main__':
    args = parser.visualizer_parser()
    # Sample data (replace with your actual data)
    #visualize_test_results(args=args)
    dataset = HierarchyDataset(annotation_file_path=args.test_file,path_to_model=None,image_count_threshold=-1, hierarchy_file_path=args.hierarchy_file,image_dir=args.image_dir, hierarchy_dicts_file_path =args.hierarchy_dicts_file,hierarchy_depth=args.hierarchy_depth)
    random.seed(42)
    os.makedirs(args.output_dir,exist_ok=True)
    # Checks if GPU Support ist active
    device = torch.device("cuda") if args.gpu else torch.device("cpu")
    hierarchy_dicts = dataset.filtered_hierarchy_dicts
    num_classes_list = dh.get_num_classes_from_hierarchy(hierarchy_dicts)
    explicit_hierarchy = torch.tensor(dh.generate_hierarchy_matrix_from_tree(hierarchy_dicts)).to(device=device)
    random_indexes = random.sample(range(len(dataset.image_label_tuple_list)), args.image_count)
    for i in random_indexes:
        image_file_path = dataset.image_label_tuple_list[i][0]
        true_label = dataset.image_label_tuple_list[i][1]
        
       
        
        output_file_path = os.path.join(args.output_dir,f'sample_image{i}.png')
        visualize_sample_image(image_file_path=image_file_path,true_label=true_label,model_names=args.model_names, best_model_dirs=args.model_dirs,threshold=0.5,hierarchy_dicts=hierarchy_dicts,output_file_path=output_file_path,explicit_hierarchy=explicit_hierarchy,num_classes_list=num_classes_list,device=device)
    
    