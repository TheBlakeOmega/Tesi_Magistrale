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
        data_loader = self._getDataLoader(data, data.train_mask, batch_size=64)
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
            for batch in data_loader:
                batch_example_indexes = batch[0]
                batch_train_mask = batch[1]

                optimizer.zero_grad()
                out = self.model(data, batch_example_indexes)
                loss = criterion(out[batch_train_mask], data.y[batch_example_indexes][batch_train_mask].squeeze())
                total_loss += float(loss)
                acc += accuracy_score(data.y[batch_example_indexes][batch_train_mask].squeeze().tolist(),
                                      out[batch_train_mask].argmax(dim=1).tolist())
                loss.backward()
                optimizer.step()

            # Validation
            if data.validation_mask.sum() > 0:
                with torch.no_grad():
                    self.model.eval()
                    out = self.model(data)
                    val_loss = float(criterion(out[data.validation_mask],
                                               data.y[data.validation_mask].squeeze()))
                    val_acc = accuracy_score(data.y[data.validation_mask].squeeze().tolist(),
                                             out[data.validation_mask].argmax(dim=1).tolist())

            # Print metrics
            print(f'TRM: Epoch {epoch + 1:>3} | Train Loss: {total_loss / len(data_loader):.3f} '
                  f'| Train Acc: {acc / len(data_loader):.3f} | Val Loss: '
                  f'{val_loss:.3f} | Val Acc: {val_acc:.3f}')

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

    def test(self, input_graph, device, mask=None):
        """
        Train model with input torch_geometric graph
        :param input_graph:
        graph on which compute predictions
        :param device:
        device in which computation will be executed
        :param mask:
        default None. A tensor of boolean that will be used to select a predictions' subset
        :return:
        A tensor with predictions
        """
        self.model.eval()
        predictions = self.model(input_graph.to(device))
        if mask is not None:
            predictions = predictions[mask]
        predictions = predictions.argmax(dim=1)
        return predictions

    def loadModel(self, load_path, device):
        """
        Loads a model from storage
        :param load_path:
        path from which load the model
        :param device:
        device in which model will be used
        """
        with open(load_path, 'rb') as file:
            print("LM: Load model in pickle object")
            self.model = pickle.load(file)
            self.model.to(device)
            print("LM: Model loaded")
            file.close()

    def _getDataLoader(self, data, masks, batch_size=64):
        """
        Builds a dataloader from torch geometric library
        :param data:
        input graph
        :param masks:
        tensor of boolean used as data mask
        :param batch_size:
        default at 64. Size of batches created in result loader
        :return:
        torch_geometric.loader.DataLoader instance, that uses a random sampler
        """
        tensorDataset = TensorDataset(torch.tensor(list(range(len(data.x))), dtype=torch.int), masks)
        random_sampler = RandomSampler(tensorDataset)
        data_loader = DataLoader(tensorDataset, sampler=random_sampler, batch_size=batch_size)
        return data_loader


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
        x = torch_functional.dropout(x, p=0.3)
        x = self.conv2(x, edge_index, edge_weights)
        x = torch_functional.relu(x)
        x = torch_functional.dropout(x, p=0.3)

        x = self.lin1(x)
        x = torch_functional.relu(x)
        x = self.lin2(x)
        x = torch_functional.relu(x)
        x = self.lin3(x)

        if batch_node_indexes is not None:
            x = x[batch_node_indexes]

        x = torch_functional.softmax(x, dim=1)
        return x
