import torch
import torch.nn as nn

class ODEFunc(nn.Module):
    def __init__(self, n_nodes):
        super(ODEFunc, self).__init__()
        self.fc1 = nn.Linear(n_nodes * n_nodes, 512)
        self.fc2 = nn.Linear(512, n_nodes * n_nodes)

    def forward(self, t, A_flat):
        A_feat = torch.relu(self.fc1(A_flat))
        dA_dt = self.fc2(A_feat)
        return dA_dt
