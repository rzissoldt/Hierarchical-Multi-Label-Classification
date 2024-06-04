import torch, sys
from torchmetrics.classification import Precision, Recall, F1Score
# Define test data
binary_predictions = torch.tensor([
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0]
])

labels = torch.tensor([
    [1, 0, 1],
    [0, 1, 0],
    [1, 1, 1],
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0],
    [1, 0, 1],
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 1],
    [0, 0, 1],
    [0, 1, 1],
    [0, 1, 0],
    [1, 0, 0],
    [1, 0, 1],
    [0, 0, 1],
    [0, 1, 1],
    [0, 1, 0],
    [1, 0, 0]
])
def precision_recall_f1_score(binary_predictions, labels, average='micro'):
    """
    Calculate precision, recall, and F1 score for multi-class classification.
    
    Args:
    - binary_predictions (torch.Tensor): Predicted probabilities (shape: (n, num_classes)).
    - labels (torch.Tensor): Ground truth labels (shape: (n, num_classes)).
    - average (str): Type of averaging to perform ('micro' or 'macro').
    
    Returns:
    - precision (float): Precision score.
    - recall (float): Recall score.
    - f1_score (float): F1 score.
    """
    
    # True Positives, False Positives, False Negatives
    TP = torch.sum((binary_predictions == 1) & (labels == 1), dim=0).float()
    FP = torch.sum((binary_predictions == 1) & (labels == 0), dim=0).float()
    FN = torch.sum((binary_predictions == 0) & (labels == 1), dim=0).float()
    
    # Calculate precision, recall, and F1 score for each class
    precision = TP / (TP + FP + 1e-10)  # Adding epsilon to avoid division by zero
    recall = TP / (TP + FN + 1e-10)  # Adding epsilon to avoid division by zero
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)  # Adding epsilon to avoid division by zero
    
    # Perform averaging
    if average == 'micro':
        precision = torch.sum(TP) / (torch.sum(TP) + torch.sum(FP) + 1e-10)
        recall = torch.sum(TP) / (torch.sum(TP) + torch.sum(FN) + 1e-10)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
    elif average == 'macro':
        precision = torch.mean(precision)
        recall = torch.mean(recall)
        f1_score = torch.mean(f1_score)
    else:
        raise ValueError("Invalid value for 'average'. Allowed values are 'micro' or 'macro'.")
    
    return precision.item(), recall.item(), f1_score.item()
# Test the function with 'micro' averaging
precision_micro, recall_micro, f1_score_micro = precision_recall_f1_score(binary_predictions, labels, average='micro')
print(f"Micro Averaging - Precision: {precision_micro}, Recall: {recall_micro}, F1 Score: {f1_score_micro}")

# Test the function with 'macro' averaging
precision_macro, recall_macro, f1_score_macro = precision_recall_f1_score(binary_predictions, labels, average='macro')
print(f"Macro Averaging - Precision: {precision_macro}, Recall: {recall_macro}, F1 Score: {f1_score_macro}")

precision = Precision(task="multilabel", average='micro', num_labels=3, threshold=0.5)
torch_precision_micro = precision(binary_predictions, labels)
recall = Recall(task="multilabel", average='micro', num_labels=3, threshold=0.5)
torch_recall_micro = recall(binary_predictions, labels)
f1_score = F1Score(task="multilabel", average ='micro', num_labels=3, threshold= 0.5)
torch_f1_micro = f1_score(binary_predictions, labels)
print(f"Torchmetrics Micro Averaging - Precision: {torch_precision_micro.item()}, Recall: {torch_recall_micro.item()}, F1 Score: {torch_f1_micro.item()}")
precision = Precision(task="multilabel", average='macro', num_labels=3, threshold=0.5)
torch_precision_macro = precision(binary_predictions, labels)
recall = Recall(task="multilabel", average='macro', num_labels=3, threshold=0.5)
torch_recall_macro = recall(binary_predictions, labels)
f1_score = F1Score(task="multilabel", average ='macro', num_labels=3, threshold= 0.5)
torch_f1_macro = f1_score(binary_predictions, labels)
print(f"Torchmetrics Macro Averaging - Precision: {torch_precision_macro.item()}, Recall: {torch_recall_macro.item()}, F1 Score: {torch_f1_macro.item()}")
