import torch

# Check if CUDA is available
if torch.cuda.is_available():
    # Print the name of the current CUDA device
    print(f"CUDA is available. Device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available.")