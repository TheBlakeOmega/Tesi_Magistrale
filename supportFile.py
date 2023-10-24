import torch
import torch_directml
from torch_geometric.data import Data

dml = torch_directml.device()

x = torch.tensor([[1], [2], [3], [4], [5]]).to(dml)
edges = torch.tensor([[1, 2], [1, 2], [3, 4], [2, 1]]).to(dml)

graph = Data(x=x, edge_index=edges)
print(str(graph.num_nodes))
print(str(graph.num_edges))
print(str(graph.is_directed()))
