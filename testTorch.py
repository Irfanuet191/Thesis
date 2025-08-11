import torch

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create tensors and move them to the GPU
a = torch.rand(3, 3).to(device)
b = torch.rand(3, 3).to(device)

# Perform a simple operation on GPU
c = a + b

print("Result on GPU:")
print(c)
