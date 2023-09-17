import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d as BN
from torch.nn import Linear, ReLU, Sequential

from torch_geometric.nn import (GINConv, JumpingKnowledge,
                                global_mean_pool, SAGEConv, GCNConv,
                                TopKPooling, ASAPooling, GATConv)



class Propose(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, ratio=0.8, heads=8):
        super().__init__()
        self.conv1 = GATConv(dataset.num_features, hidden, heads, dropout=0.6)
        self.conv2 = GATConv(hidden * heads, hidden, heads=1, dropout=0.6)
        self.pool1 = TopKPooling(hidden, ratio)
        self.pool2 = TopKPooling(hidden, ratio)
        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.pool1.reset_parameters()
        self.pool2.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index,
                                                   batch=batch)
        x = global_mean_pool(x, batch)
        # x = F.relu(self.lin1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)



    def __repr__(self):
        return self.__class__.__name__

class ProposeTF(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, ratio=0.8, heads=8):
        super().__init__()
        self.conv1 = GATConv(dataset.num_features, hidden, heads, dropout=0.6)
        self.conv2 = GATConv(hidden * heads, hidden, heads=1, dropout=0.6)
        self.pool1 = TopKPooling(hidden, ratio)
        self.pool2 = TopKPooling(hidden, ratio)
        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.pool1.reset_parameters()
        self.pool2.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        # x, edge_index, _, batch, _, _ = self.pool2(x, edge_index,
        #                                            batch=batch)
        x = global_mean_pool(x, batch)
        # x = F.relu(self.lin1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)


class ProposeT(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, ratio=0.8, heads=8):
        super().__init__()
        self.conv1 = GATConv(dataset.num_features, hidden, heads, dropout=0.6)
        self.conv2 = GATConv(hidden * heads, hidden, heads=1, dropout=0.6)
        self.pool1 = TopKPooling(hidden, ratio)
        self.pool2 = TopKPooling(hidden, ratio)
        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.pool1.reset_parameters()
        self.pool2.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        # x, edge_index, _, batch, _, _ = self.pool2(x, edge_index,
        #                                            batch=batch)
        x = global_mean_pool(x, batch)
        # x = F.relu(self.lin1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)


class ProposeF(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, ratio=0.8, heads=8):
        super().__init__()
        self.conv1 = GATConv(dataset.num_features, hidden, heads, dropout=0.6)
        self.conv2 = GATConv(hidden * heads, hidden, heads=1, dropout=0.6)
        self.pool1 = TopKPooling(hidden, ratio)
        self.pool2 = TopKPooling(hidden, ratio)
        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.pool1.reset_parameters()
        self.pool2.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index,
                                                   batch=batch)
        x = global_mean_pool(x, batch)
        # x = F.relu(self.lin1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)



    def __repr__(self):
        return self.__class__.__name__
