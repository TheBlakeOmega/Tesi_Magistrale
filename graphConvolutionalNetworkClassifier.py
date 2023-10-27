import torch
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import torch.nn.functional as F


class GCNClassifier:

    def __init__(self, configuration, dataset_configuration):
        self.configuration = configuration
        self.ds_configuration = dataset_configuration
        self.model = None






class GCN(torch.nn.Module):
    """
    Class to obtain a GCN model for homogeneous graphs
    """
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 512)

        self.lin1 = Linear(512, 64)
        self.lin2 = Linear(64, 32)
        self.lin3 = Linear(32, num_classes)

        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

    def forward(self, data, batch=None):
        x, edge_index, edge_weights = data.x, data.edge_index, data.edge_weight

        x = self.conv1(x, edge_index, edge_weights)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.5)

        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)
        x = self.lin3(x)

        if batch is not None:
            x = x[batch]
        x = F.softmax(x, dim=1)
        return x
