import torch
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import torch.nn.functional as torch_functional
from torch_geometric.loader import NeighborLoader
from torch.nn import CrossEntropyLoss
import pickle
import numpy as np
from datetime import datetime
from torch.utils.data import TensorDataset, RandomSampler, DataLoader


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
        start_train_time = np.datetime64(datetime.now())
        data = input_graph.to(device)
        self.model = GCN(data.num_node_features, data.num_classes).to(device)
        optimizer = torch.optim.Adam(self.model.parameters())
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=300)
        data_loader = NeighborLoader(data, num_neighbors=[10] * 2, batch_size=64, input_nodes=data.train_mask)

        self.model.train()
        criterion = CrossEntropyLoss()
        last_loss = 10
        worst_loss_times = 0
        total_loss = 0
        acc = 0
        val_loss = 0
        val_acc = 0
        for epoch in range(args['epochs']):

            # Train on batches
            loss = 0
            for batch in data_loader:
                optimizer.zero_grad()
                out = self.model(batch.x, batch.edge_index, batch.edge_weight)
                loss = criterion(out[batch.train_mask], batch.y[batch.train_mask].squeeze())
                total_loss += float(loss)
                acc += self._accuracy(out[batch.train_mask].argmax(dim=1), batch.y[batch.train_mask])
                loss.backward()
                optimizer.step()

                # Validation
                if batch.validation_mask.sum() > 0:
                    val_loss += float(criterion(out[batch.validation_mask], batch.y[batch.validation_mask].squeeze()))
                    val_acc += self._accuracy(out[batch.validation_mask].argmax(dim=1),
                                              batch.y[batch.validation_mask])

            # Print metrics
            print(f'TRM: Epoch {epoch:>3} | Train Loss: {total_loss / len(data_loader):.3f} '
                  f'| Train Acc: {acc / len(data_loader):>6.2f}% | Val Loss: '
                  f'{val_loss / len(data_loader):.3f} | Val Acc: '
                  f'{val_acc / len(data_loader):>6.2f}%')

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

    def _accuracy(self, pred_y, y):
        """Calculate accuracy."""
        return ((pred_y == y).sum() / len(y)).item()


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

    def forward(self, x, edge_index, edge_weights):

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
        x = torch_functional.softmax(x, dim=1)
        return x
