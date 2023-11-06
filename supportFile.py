import torch
from torch_geometric.data import Data
import pickle

x = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.int8).to('cpu')
edges = torch.tensor([[0, 1, 3], [4, 4, 3]], dtype=torch.int8).to('cpu')
a = torch.tensor([True, False, True, False, True, False, True, False, True, False, True, False], dtype=torch.int8).to(
    'cpu')

print((a == True).nonzero().flatten())

graph = Data(x=x, edge_index=edges)
graph.size()
# graph.put_edge_index((torch.tensor([1]).to(dml),torch.tensor([graph.num_nodes]).to(dml)))
# graph.put_tensor(torch.tensor([[6]]).to(dml))

graph.remove_edge_index(torch.tensor([[0], [4]], dtype=torch.int8).to('cpu'))

# graph.x = graph.get_tensor().put([6], graph.get_tensor().size)
print(str(graph.is_coalesced()))
print(str(graph.size()))
print(str(graph.get_tensor_size()))

# with open('results/pytorchGraphs/MALDROID_train_torch_graph.pkl', 'rb') as file:
#    graph = pickle.load(file)
#    print(str(graph.is_coalesced()))
