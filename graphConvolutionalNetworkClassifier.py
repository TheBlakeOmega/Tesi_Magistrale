import torch
from torch.utils.data import TensorDataset, RandomSampler
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import torch.nn.functional as torch_functional
from torch_geometric.loader import DataLoader
from torch.nn import CrossEntropyLoss
import pickle
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score
import pandas as pd
from torch_geometric.data import Data
import torch_geometric


class GraphNetwork:

    def __init__(self):
        self.model = None

    def train(self, input_graph, args, device, save_path=None):
        """
        Train model with input torch_geometric graph and inputs in args
        :param input_graph:
        input graph with train data
        :param args:
        parameters used during train
        :param device:
        device that will compute tensors
        :param save_path:
        default None. If specified, it will be the path in which the trained graph will be saved
        """
        
        features = torch.tensor([[10], [20], [30], [40], [50], [60]],
                                dtype=torch.float).to(device)
        labels = torch.tensor([[0], [1], [2], [2], [1], [0]], dtype=torch.long).to(device)
        edge_dataframe = pd.DataFrame([
            [0, 1, 1.0],
            [0, 2, 2.0],
            [0, 4, 3.0],
            [1, 2, 4.0],
            [1, 5, 4.0],
            [1, 3, 3.0],
            [1, 4, 2.0],
            [2, 3, 1.0],
            [3, 4, 2.0],
            [4, 5, 1.0],
        ], columns=['source', 'target', 'weight'])
        edge_index = torch.tensor([edge_dataframe['source'], edge_dataframe['target']], dtype=torch.long).to(device)
        edge_weights = torch.tensor(edge_dataframe['weight'], dtype=torch.float).to(device)
        torch_train_mask, torch_validation_mask = torch.tensor([True, True, False, True, True, False],
                                                               dtype=torch.bool).to(device), torch.tensor(
            [False, False, True, False, False, True], dtype=torch.bool).to(device)
        graph = Data(x=features, y=labels, edge_index=edge_index, edge_weight=edge_weights,
                     num_classes=3, num_nodes=6).to(device)
        graph = graph.coalesce()
        graph.train_mask = torch_train_mask
        graph.validation_mask = torch_validation_mask
        input_graph = torch_geometric.transforms.ToUndirected()(graph)

        start_train_time = np.datetime64(datetime.now())
        data = input_graph.to(device)
        self.model = GCN(data.num_node_features, data.num_classes).to(device)
        optimizer = torch.optim.Adam(self.model.parameters())
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=300)
        # data_loader = NeighborLoader(data, num_neighbors=[-1], batch_size=64, input_nodes=data.train_mask)
        data_loader = self._getDataLoader(data)
        print("TRM: DataLoader created with " + str(len(data_loader)) + " batches")

        criterion = CrossEntropyLoss()
        last_loss = 10
        worst_loss_times = 0
        for epoch in range(args['epochs']):

            # Train on batches
            loss = 0
            acc = 0
            val_loss = 0
            val_acc = 0
            total_loss = 0
            self.model.train()
            print("TRM: Start train epoch " + str(epoch + 1))
            for batch in data_loader:
                batch_example_indexes = batch[0]
                batch_train_mask = batch[1]

                optimizer.zero_grad()
                out = self.model(data, batch_example_indexes)
                loss = criterion(out[batch_train_mask], data.y[batch_example_indexes][batch_train_mask].squeeze())
                total_loss += float(loss)
                acc += accuracy_score(data.y[batch_example_indexes][batch_train_mask].squeeze().tolist(),
                                      out[batch_train_mask].argmax(dim=1).tolist())
                #print(data.y[batch_example_indexes][batch_train_mask].squeeze().tolist())
                #print(out[batch_train_mask].argmax(dim=1).tolist())
                loss.backward()
                optimizer.step()

            # Validation
            print("TRM: Start validation epoch " + str(epoch + 1))
            if data.validation_mask.sum() > 0:
                with torch.no_grad():
                    self.model.eval()
                    out = self.model(data)
                    val_loss += float(criterion(out[data.validation_mask],
                                                data.y[data.validation_mask].squeeze()))
                    val_acc += accuracy_score(data.y[data.validation_mask].squeeze().tolist(),
                                              out[data.validation_mask].argmax(dim=1).tolist())

            # Print metrics
            print(f'TRM: Epoch {epoch + 1:>3} | Train Loss: {total_loss / len(data_loader):.3f} '
                  f'| Train Acc: {acc / len(data_loader):.3f} | Val Loss: '
                  f'{val_loss / data.validation_mask.sum():.3f} | Val Acc: '
                  f'{val_acc / data.validation_mask.sum():.3f}')

            scheduler.step()

            if loss > last_loss:
                worst_loss_times += 1

            last_loss = loss

            if worst_loss_times == args['earlyStoppingThresh']:
                break

        end_train_time = np.datetime64(datetime.now())

        if save_path is not None:
            with open(save_path, 'wb') as file:
                print("TRM: Saving model in pickle object")
                pickle.dump(self.model, file)
                print("TRM: Model saved")
                file.close()

        return total_loss / len(data_loader), end_train_time - start_train_time

    def _getDataLoader(self, data):
        tensorDataset = TensorDataset(torch.tensor(list(range(len(data.x))), dtype=torch.int), data.train_mask)
        random_sampler = RandomSampler(tensorDataset)
        data_loader = DataLoader(tensorDataset, sampler=random_sampler, batch_size=2)
        return data_loader

    def _computeBatch(self, data, batch_node_indexes):
        x, edge_index, edge_weights = data.x, data.edge_index, data.edge_weight
        edge_index = edge_index.tolist()
        batch_node_indexes = batch_node_indexes.tolist()
        print(batch_node_indexes)
        print(edge_index)
        edge_index = [pair for pair in edge_index if pair[0] in batch_node_indexes]
        print(edge_index)


class GCN(torch.nn.Module):
    """
    Class to obtain a GCN model for homogeneous graphs
    """

    def __init__(self, num_node_features, num_classes):
        super().__init__()
        torch.manual_seed(12)
        self.conv1 = GCNConv(num_node_features, 512)
        self.conv2 = GCNConv(512, 256)

        self.lin1 = Linear(256, 128)
        self.lin2 = Linear(128, 64)
        self.lin3 = Linear(64, num_classes)

    def forward(self, data, batch_node_indexes=None):
        x, edge_index, edge_weights = data.x, data.edge_index, data.edge_weight

        x = self.conv1(x, edge_index, edge_weights)
        x = torch_functional.relu(x)
        x = torch_functional.dropout(x, training=self.training, p=0.3)
        x = self.conv2(x, edge_index, edge_weights)
        x = torch_functional.relu(x)
        x = torch_functional.dropout(x, training=self.training, p=0.3)

        x = self.lin1(x)
        x = torch_functional.relu(x)
        x = self.lin2(x)
        x = torch_functional.relu(x)
        x = self.lin3(x)

        if batch_node_indexes is not None:
            x = x[batch_node_indexes]

        x = torch_functional.softmax(x, dim=1)
        return x
