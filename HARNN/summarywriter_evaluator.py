from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
import numpy as np
import matplotlib.pyplot as plt
import sys
import os, json
from types import SimpleNamespace
# Add the parent directory to the Python path
sys.path.append('../')
from utils import xtree_utils as xtree
from utils import data_helpers as dh
from utils import param_parser as parser
def analyze_summarywriter_dir(dir):
    def get_subdirectories(directory):
        subdirectories = [os.path.join(directory, name) for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
        return subdirectories

    model_dirs = get_subdirectories(dir)
    
    best_model_index = 0
    best_auprc_score = 0.
    best_model_config= None
    model_list = []
    for i in range(len(model_dirs)):
        model_dir = model_dirs[i]
        model_metric, model_config = get_metric_from_dir(model_dir)
        if 'AveragePrecision' in model_metric:
            if model_metric['AveragePrecision'] > best_auprc_score:
                best_auprc_score = model_metric['AveragePrecision']
                model_list.append(
                    {
                        'model_dir':model_dir,
                        'average_precision':model_metric['AveragePrecision']
                    }
                )

    # Sort the list of dictionaries based on the 'age' key in descending order
    sorted_list = sorted(model_list, key=lambda x: x['average_precision'], reverse=True)
    best_model_dir = model_dirs[best_model_index]
    with open(os.path.join(best_model_dir,'model_config.json')) as infile:
        best_model_config = SimpleNamespace(**json.load(infile))
    print("Top 5 Models:", sorted_list[:5])
    print(f'Model Config:{best_model_config} from {best_model_dir}')
    print(f'Best AveragePrecision Score was {best_auprc_score}')
    best_model_file_path = os.path.join(best_model_dir,'models',os.listdir(os.path.join(best_model_dir,'models'))[0])
    return best_model_file_path, best_model_config



def extract_file_with_prefix(directory, prefix):
    # Get the list of files in the directory
    files = os.listdir(directory)
    
    # Filter files that start with the specified prefix
    matching_files = [file for file in files if file.startswith(prefix)]
    
    return matching_files

def get_config_and_event_file_from_dir(model_dir):
    event_file_prefix = 'events.out.tfevents'
    model_config_file = 'model_config.json'
    matching_event_file = extract_file_with_prefix(model_dir,event_file_prefix)[0]
    event_file_path = os.path.join(model_dir, matching_event_file)
    #train_loss, val_loss = get_train_val_loss_from_event_file(event_file_path)
    model_config_file_path = os.path.join(model_dir, model_config_file)
    
    return event_file_path,model_config_file_path
def get_metric_from_dir(model_dir):
    event_file_path,model_config_file_path=get_config_and_event_file_from_dir(model_dir=model_dir)
    
    with open(model_config_file_path,'r') as infile:
        model_config = json.load(infile)
    hierarchy = xtree.load_xtree_json(model_config['hierarchy_file'])
    hierarchy_dicts = xtree.generate_dicts_per_level(hierarchy)
    num_classes_list = dh.get_num_classes_from_hierarchy(hierarchy_dicts)
    metric = {}
    hierarchy_depth = len(num_classes_list)
    loader = EventFileLoader(event_file_path)
    for event in loader.Load():
        summary_value = event.summary.value
        if len(summary_value) == 0:
            continue
        
        summary_value = summary_value[0]
        # Check if the attribute 'summary' exists
        if hasattr(summary_value, 'tag'):
            if summary_value.tag == 'Validation/AverageAUC':
                if hasattr(summary_value, 'tensor'):
                    float_val = summary_value.tensor.float_val[0]
                    metric['AverageAUC'] = float_val
            if summary_value.tag == 'Validation/AveragePrecision':
                if hasattr(summary_value, 'tensor'):
                    float_val = summary_value.tensor.float_val[0]
                    metric['AveragePrecision'] = float_val
            if summary_value.tag == 'Validation/Precision':
                if hasattr(summary_value, 'tensor'):
                    float_val = summary_value.tensor.float_val[0]
                    metric['Precision'] = float_val
            if summary_value.tag == 'Validation/Recall':
                if hasattr(summary_value, 'tensor'):
                    float_val = summary_value.tensor.float_val[0]
                    metric['Recall'] = float_val
            if summary_value.tag == 'Validation/F1':
                if hasattr(summary_value, 'tensor'):
                    float_val = summary_value.tensor.float_val[0]
                    metric['F1'] = float_val
            if summary_value.tag == 'Validation/EMR':
                if hasattr(summary_value, 'tensor'):
                    float_val = summary_value.tensor.float_val[0]
                    metric['EMR'] = float_val
            for i in range(5):
                if summary_value.tag == f'Validation/PrecisionTopK/{i}':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'PrecisionTopK/{i+1}'] = float_val
                if summary_value.tag == f'Validation/RecallTopK/{i}':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'RecallTopK/{i+1}'] = float_val
                if summary_value.tag == f'Validation/F1TopK/{i}':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'F1TopK/{i+1}'] = float_val
                if summary_value.tag == f'Validation/EMRTopK/{i}':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'EMRTopK/{i+1}'] = float_val
            for i in range(hierarchy_depth):
                if summary_value.tag == f'Validation/{i+1}-LayerPrecision':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'{i+1}-LayerPrecision'] = float_val
                if summary_value.tag == f'Validation/{i+1}-LayerRecall':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'{i+1}-LayerRecall'] = float_val
                if summary_value.tag == f'Validation/{i+1}-LayerF1':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'{i+1}-LayerAUC'] = float_val
                if summary_value.tag == f'Validation/{i+1}-LayerAUC':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'{i+1}-LayerAUPRC'] = float_val
                if summary_value.tag == f'Validation/{i+1}-LayerAUPRC':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'{i+1}-LayerAUPRC'] = float_val
                if summary_value.tag == f'Validation/{i+1}-LayerPCPPrecision':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'{i+1}-LayerPCPPrecision'] = float_val
                if summary_value.tag == f'Validation/{i+1}-LayerPCPRecall':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'{i+1}-LayerPCPRecall'] = float_val
                if summary_value.tag == f'Validation/{i+1}-LayerPCPF1':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'{i+1}-LayerPCPF1'] = float_val
                if summary_value.tag == f'Validation/{i+1}-LayerPCPAUC':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'{i+1}-LayerPCPAUC'] = float_val
                if summary_value.tag == f'Validation/{i+1}-LayerPCPAUPRC':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'{i+1}-LayerPCPAUPRC'] = float_val
                        
            if summary_value.tag == 'Validation/PCPAverageAUC':
                if hasattr(summary_value, 'tensor'):
                    float_val = summary_value.tensor.float_val[0]
                    metric['PCPAverageAUC'] = float_val
            if summary_value.tag == 'Validation/PCPAveragePrecision':
                if hasattr(summary_value, 'tensor'):
                    float_val = summary_value.tensor.float_val[0]
                    metric['PCPAveragePrecision'] = float_val
            if summary_value.tag == 'Validation/PCPPrecision':
                if hasattr(summary_value, 'tensor'):
                    float_val = summary_value.tensor.float_val[0]
                    metric['PCPPrecision'] = float_val
            if summary_value.tag == 'Validation/PCPRecall':
                if hasattr(summary_value, 'tensor'):
                    float_val = summary_value.tensor.float_val[0]
                    metric['PCPRecall'] = float_val
            if summary_value.tag == 'Validation/PCPF1':
                if hasattr(summary_value, 'tensor'):
                    float_val = summary_value.tensor.float_val[0]
                    metric['PCPF1'] = float_val
            if summary_value.tag == 'Validation/PCPEMR':
                if hasattr(summary_value, 'tensor'):
                    float_val = summary_value.tensor.float_val[0]
                    metric['PCPEMR'] = float_val
            for i in range(5):
                if summary_value.tag == f'Validation/PCPPrecisionTopK/{i}':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'PCPPrecisionTopK/{i+1}'] = float_val
                if summary_value.tag == f'Validation/PCPRecallTopK/{i}':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'PCPRecallTopK/{i+1}'] = float_val
                if summary_value.tag == f'Validation/PCPF1TopK/{i}':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'PCPF1TopK/{i+1}'] = float_val
                if summary_value.tag == f'Validation/PCPEMRTopK/{i}':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'PCPEMRTopK/{i+1}'] = float_val
                        
    return metric, model_config
                
def get_train_val_loss_from_event_file(event_file):
    train_loss = []
    val_loss = []
    loader = EventFileLoader(event_file)
    for event in loader.Load():
        summary_value = event.summary.value
        if len(summary_value) == 0:
            continue
        
        summary_value = summary_value[0]
        # Check if the attribute 'summary' exists
        if hasattr(summary_value, 'tag'):
            if summary_value.tag == 'Training/Loss':
                if hasattr(summary_value, 'tensor'):
                    tensor = summary_value.tensor
                    if hasattr(tensor, 'float_val'):
                        train_loss.append(tensor.float_val[0])
            if summary_value.tag == 'Validation/Loss':
                if hasattr(summary_value, 'tensor'):
                    tensor = summary_value.tensor
                    if hasattr(tensor, 'float_val'):
                        val_loss.append(tensor.float_val[0])
    
    return np.array(train_loss), np.array(val_loss)

def plot_train_val_loss_from_event_file(event_file):
    train_loss, val_loss = get_train_val_loss_from_event_file(event_file=event_file)
    
    # Determine the ratio between the lengths of the two lists
    ratio = len(val_loss) / len(train_loss)

    # Initialize a new list to store averaged validation loss values
    new_val_loss = []

    # Iterate over the training loss values
    for i in range(len(train_loss)):
        # Determine the start and end indices for averaging
        start_index = int(i * ratio)
        end_index = int((i + 1) * ratio)
        if start_index-end_index == 0:
            new_val_loss.append(val_loss[start_index])
        else:
            # Calculate the average of validation loss values within the range
            avg_val_loss = np.mean(val_loss[start_index:end_index])
            new_val_loss.append(avg_val_loss)

    # Now new_val_loss has the same length as train_loss
    
    # Plotting
    batch_steps = np.arange(len(train_loss))  # Assuming both train_loss and new_val_loss have the same length
    
    plt.plot(batch_steps, train_loss, label='Train Loss')
    plt.plot(batch_steps, new_val_loss, label='Validation Loss')
        
    # Axis labels and legend
    plt.xlabel('Batch Steps')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()

if __name__ == '__main__':
    args = parser.evaluator_parser()
    analyze_summarywriter_dir(model_dir)
        
        
    
    