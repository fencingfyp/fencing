import torch

# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device.")
else:
    device = torch.device("cpu")
    print("MPS not available, using CPU.")

# Create a tensor and move it to the MPS device
x = torch.rand(5, 5).to(device)
print(x)