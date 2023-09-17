import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d as BN
from torch.nn import Linear, ReLU, Sequential

import torch.nn as nn

from torch_geometric.nn import global_add_pool
from torch_geometric.nn import GINConv,GATConv,BatchNorm
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden)
        self.conv2 = GCNConv(hidden, dataset.num_classes)


    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class GIN(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden):
        super().__init__()
        self.conv1 = GINConv(
            Sequential(
                Linear(dataset.num_features, hidden),
                ReLU(),
                Linear(hidden, hidden),
                ReLU(),
                BN(hidden),
            ), train_eps=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        Linear(hidden, hidden),
                        ReLU(),
                        BN(hidden),
                    ), train_eps=True))
        self.lin1 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.lin1(x)
        return F.log_softmax(x, dim=-1)
    def __repr__(self):
        return self.__class__.__name__




class GAT(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, heads=8):
        super().__init__()
        self.conv1 = GATConv(dataset.num_features, hidden, heads, dropout=0.6)
        self.conv2 = GATConv(hidden * heads, dataset.num_classes, heads=1, dropout=0.6)


    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__




class GraphSAGE(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden):
        super().__init__()
        self.conv1 = SAGEConv(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden, hidden))
        self.convs.append(SAGEConv(hidden, dataset.num_classes))


    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = global_mean_pool(x, batch)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class EEGGraphConvNet(nn.Module):
    def __init__(self,  dataset, num_layers, hidden, sfreq=None, batch_size=32):
        super(EEGGraphConvNet, self).__init__()
        # layers
        self.conv1 = GCNConv(dataset.num_features, 32, improved=True, cached=True, normalize=False)
        self.conv2 = GCNConv(32, 20, improved=True, cached=True, normalize=False)
        self.conv2_bn = BatchNorm(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.fc_block1 = nn.Linear(20, 10)
        self.fc_block2 = nn.Linear(10, 2)
        # Xavier initializations  #init gcn layers

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

        self.fc_block1.reset_parameters()
        self.fc_block2.reset_parameters()


    # def forward(self, x, edge_index, edge_weight, batch, return_graph_embedding=False):
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.leaky_relu(self.conv2_bn(self.conv2(x, edge_index)))
        out = global_add_pool(x, batch=batch)
        out = F.dropout(out, p=0.2, training=self.training)
        out = F.leaky_relu(self.fc_block1(out))
        out = self.fc_block2(out)
        return F.log_softmax(out, dim=-1)



class EEGGraphConvNetDeep(nn.Module):
    def __init__(self, dataset, num_layers, hidden, sfreq=None, batch_size=32):
        super(EEGGraphConvNetDeep, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 16, improved=True, cached=True, normalize=False)
        self.conv2 = GCNConv(16, 32, improved=True, cached=True, normalize=False)
        self.conv3 = GCNConv(32, 64, improved=True, cached=True, normalize=False)
        self.conv4 = GCNConv(64, 50, improved=True, cached=True, normalize=False)
        self.conv4_bn = BatchNorm(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fc_block1 = nn.Linear(50, 30)
        self.fc_block2 = nn.Linear(30, 20)
        self.fc_block3 = nn.Linear(20, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.conv4.reset_parameters()
        self.fc_block1.reset_parameters()
        self.fc_block2.reset_parameters()
        self.fc_block3.reset_parameters()

    # def forward(self, x, edge_index, edge_weight, batch, return_graph_embedding=False):
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.leaky_relu(self.conv1(x, edge_index))

        x = F.leaky_relu(self.conv2(x, edge_index))

        x = F.leaky_relu(self.conv3(x, edge_index))
        x = F.leaky_relu(self.conv4_bn(self.conv4(x, edge_index)))
        out = global_add_pool(x, batch=batch)
        out = F.leaky_relu(self.fc_block1(out), negative_slope=0.01)
        out = F.dropout(out, p=0.2, training=self.training)
        out = F.leaky_relu(self.fc_block2(out), negative_slope=0.01)
        out = self.fc_block3(out)

        return F.log_softmax(out, dim=-1)