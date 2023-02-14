import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,in_dim,hidden_dim,out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,out_dim),
            nn.Softmax()
        )

    def forward(self,x,b_s):
        x = torch.reshape(x,(x.shape[0],x.shape[1]*x.shape[2]))
        out = self.mlp(x)
        return out

# to try 
# class LSTM()
# class TRANSFORMER()

