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
                best_model_index = i

    # Sort the list of dictionaries based on the 'age' key in descending order
    sorted_list = sorted(model_list, key=lambda x: x['average_precision'], reverse=True)
    best_model_dir = model_dirs[best_model_index]
    print('Best model dir:',best_model_dir)
    with open(os.path.join(best_model_dir,'model_config.json')) as infile:
        best_model_config = SimpleNamespace(**json.load(infile))
    print("Top 5 Models:", sorted_list[:5])
    print(f'Model Config:{best_model_config} from {best_model_dir}')
    print(f'Best AveragePrecision Score was {best_auprc_score}')
    best_model_file_path = os.path.join(best_model_dir,'models',os.listdir(os.path.join(best_model_dir,'models'))[0])
    print('Best Model file path:', best_model_file_path)
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
            if summary_value.tag == 'Validation/MacroAverageAUC':
                if hasattr(summary_value, 'tensor'):
                    float_val = summary_value.tensor.float_val[0]
                    metric['MacroAverageAUC'] = float_val
            if summary_value.tag == 'Validation/MacroAveragePrecision':
                if hasattr(summary_value, 'tensor'):
                    float_val = summary_value.tensor.float_val[0]
                    metric['MacroAveragePrecision'] = float_val
            if summary_value.tag == 'Validation/MacroPrecision':
                if hasattr(summary_value, 'tensor'):
                    float_val = summary_value.tensor.float_val[0]
                    metric['MacroPrecision'] = float_val
            if summary_value.tag == 'Validation/MacroRecall':
                if hasattr(summary_value, 'tensor'):
                    float_val = summary_value.tensor.float_val[0]
                    metric['MacroRecall'] = float_val
            if summary_value.tag == 'Validation/MacroF1':
                if hasattr(summary_value, 'tensor'):
                    float_val = summary_value.tensor.float_val[0]
                    metric['MacroF1'] = float_val
            if summary_value.tag == 'Validation/EMR':
                if hasattr(summary_value, 'tensor'):
                    float_val = summary_value.tensor.float_val[0]
                    metric['EMR'] = float_val
            if summary_value.tag == 'Validation/MicroAverageAUC':
                if hasattr(summary_value, 'tensor'):
                    float_val = summary_value.tensor.float_val[0]
                    metric['MicroAverageAUC'] = float_val
            if summary_value.tag == 'Validation/MicroAveragePrecision':
                if hasattr(summary_value, 'tensor'):
                    float_val = summary_value.tensor.float_val[0]
                    metric['MicroAveragePrecision'] = float_val
            if summary_value.tag == 'Validation/MicroPrecision':
                if hasattr(summary_value, 'tensor'):
                    float_val = summary_value.tensor.float_val[0]
                    metric['MicroPrecision'] = float_val
            if summary_value.tag == 'Validation/MicroRecall':
                if hasattr(summary_value, 'tensor'):
                    float_val = summary_value.tensor.float_val[0]
                    metric['MicroRecall'] = float_val
            if summary_value.tag == 'Validation/MicroF1':
                if hasattr(summary_value, 'tensor'):
                    float_val = summary_value.tensor.float_val[0]
                    metric['MicroF1'] = float_val
            if summary_value.tag == 'Validation/HierarchicalPrecision':
                if hasattr(summary_value, 'tensor'):
                    float_val = summary_value.tensor.float_val[0]
                    metric['HierarchicalPrecision'] = float_val
            if summary_value.tag == 'Validation/HierarchicalRecall':
                if hasattr(summary_value, 'tensor'):
                    float_val = summary_value.tensor.float_val[0]
                    metric['HierarchicalRecall'] = float_val
            if summary_value.tag == 'Validation/HierarchicalF1':
                if hasattr(summary_value, 'tensor'):
                    float_val = summary_value.tensor.float_val[0]
                    metric['HierarchicalF1'] = float_val
            for i in range(5):
                if summary_value.tag == f'Validation/MacroPrecisionTopK/{i}':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'MacroPrecisionTopK/{i+1}'] = float_val
                if summary_value.tag == f'Validation/MacroRecallTopK/{i}':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'MacroRecallTopK/{i+1}'] = float_val
                if summary_value.tag == f'Validation/MacroF1TopK/{i}':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'MacroF1TopK/{i+1}'] = float_val
                if summary_value.tag == f'Validation/MicroPrecisionTopK/{i}':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'MicroPrecisionTopK/{i+1}'] = float_val
                if summary_value.tag == f'Validation/MicroRecallTopK/{i}':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'MicroRecallTopK/{i+1}'] = float_val
                if summary_value.tag == f'Validation/MicroF1TopK/{i}':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'MicroF1TopK/{i+1}'] = float_val
                if summary_value.tag == f'Validation/HierarchicalPrecisionTopK/{i}':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'HierarchicalPrecisionTopK/{i+1}'] = float_val
                if summary_value.tag == f'Validation/HierarchicalRecallTopK/{i}':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'HierarchicalRecallTopK/{i+1}'] = float_val
                if summary_value.tag == f'Validation/HierarchicalF1TopK/{i}':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'HierarchicalF1TopK/{i+1}'] = float_val
                if summary_value.tag == f'Validation/EMRTopK/{i}':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'EMRTopK/{i+1}'] = float_val
            for i in range(hierarchy_depth):
                if summary_value.tag == f'Validation/{i+1}-LayerMacroPrecision':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'{i+1}-LayerMacroPrecision'] = float_val
                if summary_value.tag == f'Validation/{i+1}-LayerMacroRecall':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'{i+1}-LayerMacroRecall'] = float_val
                if summary_value.tag == f'Validation/{i+1}-LayerMacroF1':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'{i+1}-LayerMacroF1'] = float_val
                if summary_value.tag == f'Validation/{i+1}-LayerMacroAUC':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'{i+1}-LayerMacroAUC'] = float_val
                if summary_value.tag == f'Validation/{i+1}-LayerMacroAUPRC':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'{i+1}-LayerMacroAUPRC'] = float_val
                if summary_value.tag == f'Validation/{i+1}-LayerMicroPrecision':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'{i+1}-LayerMicroPrecision'] = float_val
                if summary_value.tag == f'Validation/{i+1}-LayerMicroRecall':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'{i+1}-LayerMicroRecall'] = float_val
                if summary_value.tag == f'Validation/{i+1}-LayerMicroF1':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'{i+1}-LayerMicroF1'] = float_val
                if summary_value.tag == f'Validation/{i+1}-LayerMicroAUC':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'{i+1}-LayerMicroAUC'] = float_val
                if summary_value.tag == f'Validation/{i+1}-LayerMicroAUPRC':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'{i+1}-LayerMicroAUPRC'] = float_val
                if summary_value.tag == f'Validation/{i+1}-LayerPCPMacroPrecision':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'{i+1}-LayerPCPMacroPrecision'] = float_val
                if summary_value.tag == f'Validation/{i+1}-LayerPCPMacroRecall':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'{i+1}-LayerPCPMacroRecall'] = float_val
                if summary_value.tag == f'Validation/{i+1}-LayerPCPMacroF1':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'{i+1}-LayerPCPMacroF1'] = float_val
                if summary_value.tag == f'Validation/{i+1}-LayerPCPMacroAUC':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'{i+1}-LayerPCPMacroAUC'] = float_val
                if summary_value.tag == f'Validation/{i+1}-LayerPCPMacroAUPRC':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'{i+1}-LayerPCPMacroAUPRC'] = float_val
                if summary_value.tag == f'Validation/{i+1}-LayerPCPMicroPrecision':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'{i+1}-LayerPCPMicroPrecision'] = float_val
                if summary_value.tag == f'Validation/{i+1}-LayerPCPMicroRecall':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'{i+1}-LayerPCPMicroRecall'] = float_val
                if summary_value.tag == f'Validation/{i+1}-LayerPCPMicroF1':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'{i+1}-LayerPCPMicroF1'] = float_val
                if summary_value.tag == f'Validation/{i+1}-LayerPCPMicroAUC':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'{i+1}-LayerPCPMicroAUC'] = float_val
                if summary_value.tag == f'Validation/{i+1}-LayerPCPMicroAUPRC':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'{i+1}-LayerPCPMicroAUPRC'] = float_val
                        
            if summary_value.tag == 'Validation/PCPMacroAverageAUC':
                if hasattr(summary_value, 'tensor'):
                    float_val = summary_value.tensor.float_val[0]
                    metric['PCPMacroAverageAUC'] = float_val
            if summary_value.tag == 'Validation/PCPMacroAveragePrecision':
                if hasattr(summary_value, 'tensor'):
                    float_val = summary_value.tensor.float_val[0]
                    metric['PCPMacroAveragePrecision'] = float_val
            if summary_value.tag == 'Validation/PCPMacroPrecision':
                if hasattr(summary_value, 'tensor'):
                    float_val = summary_value.tensor.float_val[0]
                    metric['PCPMacroPrecision'] = float_val
            if summary_value.tag == 'Validation/PCPMacroRecall':
                if hasattr(summary_value, 'tensor'):
                    float_val = summary_value.tensor.float_val[0]
                    metric['PCPMacroRecall'] = float_val
            if summary_value.tag == 'Validation/PCPMacroF1':
                if hasattr(summary_value, 'tensor'):
                    float_val = summary_value.tensor.float_val[0]
                    metric['PCPMacroF1'] = float_val
            if summary_value.tag == 'Validation/PCPMicroAverageAUC':
                if hasattr(summary_value, 'tensor'):
                    float_val = summary_value.tensor.float_val[0]
                    metric['PCPMicroAverageAUC'] = float_val
            if summary_value.tag == 'Validation/PCPMicroAveragePrecision':
                if hasattr(summary_value, 'tensor'):
                    float_val = summary_value.tensor.float_val[0]
                    metric['PCPMicroAveragePrecision'] = float_val
            if summary_value.tag == 'Validation/PCPMicroPrecision':
                if hasattr(summary_value, 'tensor'):
                    float_val = summary_value.tensor.float_val[0]
                    metric['PCPMicroPrecision'] = float_val
            if summary_value.tag == 'Validation/PCPMicroRecall':
                if hasattr(summary_value, 'tensor'):
                    float_val = summary_value.tensor.float_val[0]
                    metric['PCPMicroRecall'] = float_val
            if summary_value.tag == 'Validation/PCPMicroF1':
                if hasattr(summary_value, 'tensor'):
                    float_val = summary_value.tensor.float_val[0]
                    metric['PCPMicroF1'] = float_val
            if summary_value.tag == 'Validation/PCPEMR':
                if hasattr(summary_value, 'tensor'):
                    float_val = summary_value.tensor.float_val[0]
                    metric['PCPEMR'] = float_val
            for i in range(5):
                if summary_value.tag == f'Validation/PCPMacroPrecisionTopK/{i}':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'PCPMacroPrecisionTopK/{i+1}'] = float_val
                if summary_value.tag == f'Validation/PCPMacroRecallTopK/{i}':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'PCPMacroRecallTopK/{i+1}'] = float_val
                if summary_value.tag == f'Validation/PCPMacroF1TopK/{i}':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'PCPMacroF1TopK/{i+1}'] = float_val
                if summary_value.tag == f'Validation/PCPMicroPrecisionTopK/{i}':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'PCPMicroPrecisionTopK/{i+1}'] = float_val
                if summary_value.tag == f'Validation/PCPMicroRecallTopK/{i}':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'PCPMicroRecallTopK/{i+1}'] = float_val
                if summary_value.tag == f'Validation/PCPMicroF1TopK/{i}':
                    if hasattr(summary_value, 'tensor'):
                        float_val = summary_value.tensor.float_val[0]
                        metric[f'PCPMicroF1TopK/{i+1}'] = float_val
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
        
        
    
    