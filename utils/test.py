import torch
import numpy as np

# Assuming tensor has shape (128, 112, 112)
tensor = torch.randn(8,128, 112, 112)

# Calculate the total number of elements in the original tensor
print(tensor.shape[1:])
total_elements = np.prod(tensor.shape[1:])
# Calculate the size of the second dimension
second_dim_size = total_elements // 8
# Reshape the tensor
reshaped_output = tensor.view(-1,8, second_dim_size)  # -1 lets PyTorch calculate the size automatically

print(reshaped_output.shape)