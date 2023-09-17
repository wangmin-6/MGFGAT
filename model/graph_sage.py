import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.nn import SAGEConv, global_mean_pool


class GraphSAGE(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden):
        super().__init__()
        self.dim1 = 32
        self.dim2 = 32
        self.dim3 = 512
        self.conv1 = SAGEConv(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.lin1 = Linear(hidden, hidden)
        self.fc1 = torch.nn.Linear(hidden, self.dim2)
        self.fc2 = torch.nn.Linear(self.dim2, self.dim3)
        self.lin2 = Linear(self.dim3, dataset.num_classes)


    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
