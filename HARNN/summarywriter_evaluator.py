from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
import numpy as np
import matplotlib.pyplot as plt

import os, json

def analyze_summarywriter_dir(dir):
    def get_subdirectories(directory):
        subdirectories = [os.path.join(directory, name) for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
        return subdirectories

    model_dirs = get_subdirectories(dir)
    
    best_model_index = 0
    best_auc_score = 0.
    best_model_config= None
    best_train_loss = None
    best_val_loss = None
    for i in range(len(model_dirs)):
        model_dir = model_dirs[i]
        model_config, model_metric, train_loss, val_loss = get_config_and_event_file_from_dir(model_dir=model_dir)
        if model_metric['AverageAUC'] > best_auc_score:
            best_auc_score = model_metric['AverageAUC']
            best_model_index = i
            best_model_config = model_config
            best_train_loss = train_loss
            best_val_loss = val_loss
    
    print(f'Model Config:{best_model_config} from {model_dirs[i]}')
    print(f'Lowest Train Loss {min(best_train_loss)} and lowest Val Loss {min(best_val_loss)}')
    
    
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
    train_loss, val_loss = get_train_val_loss_from_event_file(event_file_path)
    with open(os.path.join(model_dir, model_config_file)) as infile:
        model_config = json.load(infile)
    model_metrics = get_pcp_metrics_from_event_file(event_file=event_file_path)
    return model_config,model_metrics, train_loss, val_loss

def get_pcp_metrics_from_event_file(event_file):
    metric = {}
    loader = EventFileLoader(event_file)
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
                        metric[f'Validation/PCPF1TopK/{i+1}'] = float_val
                        
    return metric
                
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
    
event_file_path = 'E:/workspace/Hierarchical-Multi-Label-Text-Classification/HARNN/runs/test/events.out.tfevents.1707675776.ds3.1208334.0'
model_dir = 'E:/workspace/Hierarchical-Multi-Label-Text-Classification/HARNN/runs/'
  
analyze_summarywriter_dir(model_dir)
        
        
    
    