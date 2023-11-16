from torch.nn import CrossEntropyLoss
import torch
import pickle

print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0))

x = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.int8).to('cpu')
edges = torch.tensor([[0, 1, 3], [4, 4, 3]], dtype=torch.int8).to('cpu')
a = torch.tensor([True, False, True, False, True, False, True, False, True, False, True, False], dtype=torch.int8).to(
    'cuda')

print((a == True).nonzero().flatten())

criterion = CrossEntropyLoss()
# Example of target with class indices
input = torch.randn(3, 5, requires_grad=True)
target = torch.randint(5, (3,), dtype=torch.int64)
loss = criterion(input, target)
print(input)
print(target)
print(loss)
loss.backward()
# Example of target with class probabilities
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5).softmax(dim=1)
loss = criterion(input, target)
loss.backward()
"""

graph = Data(x=x, edge_index=edges)
graph.size()
# graph.put_edge_index((torch.tensor([1]).to(dml),torch.tensor([graph.num_nodes]).to(dml)))
# graph.put_tensor(torch.tensor([[6]]).to(dml))


# graph.x = graph.get_tensor().put([6], graph.get_tensor().size)
print(str(graph.is_coalesced()))
print(str(graph.size()))

with open('results/pytorchGraphs/MALDROID_train_torch_graph.pkl', 'rb') as file:
    graph = pickle.load(file)
    print(str(graph.is_undirected()))"""
"""

# Example of target with class indices
print(torch.cuda.device_count())
loss = CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)
print(input)
print(target)
print(output)
output.backward()
# Example of target with class probabilities
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5).softmax(dim=1)
output = loss(input, target)
output.backward()"""
