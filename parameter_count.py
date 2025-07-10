import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, Linear

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_roland_parameters(input_dim, num_nodes, output_dim, hidden_conv_1=128, hidden_conv_2=128):
    """Calculate parameters for ROLANDGNN model"""
    
    # Preprocessing layers
    preprocess1_params = input_dim * 256 + 256  # Linear + bias
    preprocess2_params = 256 * 128 + 128  # Linear + bias
    
    # BatchNorm parameters (running mean, running var, weight, bias)
    bn1_params = 256 * 4  # 4 parameters per feature
    bn2_params = 128 * 4
    
    # GCN layers
    conv1_params = 128 * hidden_conv_1 + hidden_conv_1  # GCNConv + bias
    conv2_params = hidden_conv_1 * hidden_conv_2 + hidden_conv_2  # GCNConv + bias
    
    # BatchNorm for conv layers
    bn3_params = hidden_conv_1 * 4
    bn4_params = hidden_conv_2 * 4
    
    # Postprocessing layers
    postprocess1_params = hidden_conv_2 * output_dim + output_dim  # Linear + bias
    
    # NC MLP layers
    nc_mlp1_params = hidden_conv_2 * 128 + 128  # Linear + bias
    nc_mlp2_params = 128 * output_dim + output_dim  # Linear + bias
    
    # GRU cells (if used)
    gru1_params = 3 * (hidden_conv_1 * hidden_conv_1 + hidden_conv_1)  # 3 gates: input, hidden, bias
    gru2_params = 3 * (hidden_conv_2 * hidden_conv_2 + hidden_conv_2)
    
    # MLP layers for update (if used)
    mlp1_params = hidden_conv_1 * 2 * hidden_conv_1 + hidden_conv_1  # Linear + bias
    mlp2_params = hidden_conv_2 * 2 * hidden_conv_2 + hidden_conv_2  # Linear + bias
    
    # Learnable tau parameter (if used)
    tau_params = 1
    
    total_params = (preprocess1_params + preprocess2_params + 
                   bn1_params + bn2_params + bn3_params + bn4_params +
                   conv1_params + conv2_params + 
                   postprocess1_params + nc_mlp1_params + nc_mlp2_params +
                   gru1_params + gru2_params + mlp1_params + mlp2_params + tau_params)
    
    return total_params

def main():
    print("ROLANDGNN Model Parameter Analysis")
    print("=" * 50)
    
    # Based on the code analysis, here are the typical configurations:
    configurations = [
        {"name": "Link Prediction (LP) - BitcoinOTC/UCI", "input_dim": 32, "output_dim": 1, "task": "LP"},
        {"name": "Node Classification (NC) - DBLP3/DBLP5/Reddit", "input_dim": "varies", "output_dim": "varies", "task": "NC"},
        {"name": "Node Classification (NC) - SBM", "input_dim": "varies", "output_dim": "varies", "task": "NC"},
    ]
    
    # For LP tasks (fixed dimensions)
    lp_params = calculate_roland_parameters(input_dim=32, output_dim=1)
    print(f"Link Prediction Model Parameters: {lp_params:,}")
    print(f"Link Prediction Model Size: {lp_params * 4 / (1024*1024):.2f} MB (assuming float32)")
    
    # For NC tasks (example with typical dimensions)
    # Based on the code, input_dim varies by dataset, output_dim is number of classes
    nc_examples = [
        {"input_dim": 64, "output_dim": 10, "name": "NC Example 1"},
        {"input_dim": 128, "output_dim": 20, "name": "NC Example 2"},
        {"input_dim": 256, "output_dim": 50, "name": "NC Example 3"},
    ]
    
    print("\nNode Classification Model Parameters:")
    for example in nc_examples:
        params = calculate_roland_parameters(input_dim=example["input_dim"], output_dim=example["output_dim"])
        size_mb = params * 4 / (1024*1024)
        print(f"{example['name']}: {params:,} parameters ({size_mb:.2f} MB)")
    
    # Model architecture summary
    print("\nModel Architecture Summary:")
    print("- Input preprocessing: 2 Linear layers (input_dim→256→128)")
    print("- 2 GCN layers: 128→128→128")
    print("- Output layers: 128→output_dim")
    print("- BatchNorm layers after each Linear/GCN layer")
    print("- GRU cells for temporal updates (optional)")
    print("- MLP layers for temporal updates (optional)")
    
    # Size assessment
    print("\nModel Size Assessment:")
    print("- Small model: < 1M parameters")
    print("- Medium model: 1M - 10M parameters") 
    print("- Large model: 10M - 100M parameters")
    print("- Very large model: > 100M parameters")
    
    print(f"\nROLANDGNN is a {'SMALL' if lp_params < 1000000 else 'MEDIUM' if lp_params < 10000000 else 'LARGE'} model")

if __name__ == "__main__":
    main() 