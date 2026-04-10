import torch
import torch.nn as nn

class ReshapeH():
    def __init__(self, glob_shape):
        self.glob_shape = glob_shape

    def reshape_to_fill(self, h, subnodes):
        if h.shape[0] == self.glob_shape:
            return h
        
        reshaped = torch.zeros((self.glob_shape, h.shape[1]))
        reshaped[subnodes] = h

        return reshaped
    
class HopFusion(nn.Module): # Implementation of Paper (adding two hops NE together)
    def __init__(self, dim_out=16):
        super().__init__()
        self.dim_out = dim_out
        self.proj = None

    def forward(self, h0, h1):
        h_cat = torch.cat([h0, h1], dim=-1) # Shape: [1, cat_size]
        input_dim = h_cat.shape[1]

        if self.proj is None:
            self.proj = nn.Linear(input_dim, self.dim_out)
            
        return self.proj(h_cat) # Shape: [1, dim_out]
    
class MLPEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            # nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.encoder(x)