from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os, json
from types import SimpleNamespace
# Add the parent directory to the Python path
sys.path.append('../')
from utils import xtree_utils as xtree
from utils import data_helpers as dh
from utils import param_parser as parser
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

def save_hierarchy_level_metric_plot(hierarchy_level_metrics,metric_key,level,output_path):
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
    
    # Plotting
    df.plot(kind='bar')
    plt.title('Metrics per Hierarchy Level')
    plt.xlabel('Models')
    plt.ylabel(f'{metric_key}')
    plt.xticks(rotation=45)
    plot_file_path = os.path.join(output_path,f'{metric_key}.png')
    plt.savefig(plot_file_path)
    
def visualize_test_results(args):
    if len(args.result_model_dirs) != len(args.model_names):
        print('Model dirs list and model names list must have same size!')
        return
    counter = 0
    models_metric_dict = {}
    for result_model_dir in args.result_model_dirs:
        models_metric_dict[args.model_names[counter]] = get_level_metrics_from_event_file(result_model_dir)
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
    for level in range(args.hierarchy_depth):
        for metric_key in metric_keys:
            save_hierarchy_level_metric_plot(models_metric_dict,metric_key,level,args.output_dir)
if __name__ == '__main__':
    args = parser.visualizer_parser()
    # Sample data (replace with your actual data)
    visualize_test_results(args=args)