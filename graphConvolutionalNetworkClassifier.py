import torch
from torch.utils.data import TensorDataset, RandomSampler
from torch_geometric.nn import GCNConv, Linear
from torch.nn import Module, ModuleList
import torch.nn.functional as torch_functional
from torch_geometric.loader import DataLoader
from torch.nn import CrossEntropyLoss
import pickle
from hyperopt import STATUS_OK
from sklearn.metrics import accuracy_score
import numpy as np
from datetime import datetime
import csv
from torch.nn.init import xavier_uniform

SavedParameters = []
best_loss = np.inf
best_model = None


class GraphNetwork:

    def __init__(self):
        self.model = None

    def train(self, input_graph, params, device, save_path=None):
        """
        Train model with input torch_geometric graph and inputs in params
        :param input_graph:
        input graph with train data
        :param params:
        parameters used during train
        :param device:
        device that will compute tensors
        :param save_path:
        default None. If specified, it will be the path in which the trained graph will be saved
        """

        data = input_graph.to(device)
        dropouts = []
        for i in range(params['conv_layers']):
            dropouts.append(params['dropout_' + str(i + 1)])
        self.model = GCN(data.num_node_features, data.num_classes,
                         conv_layers_number=params['conv_layers'],
                         dropouts=dropouts).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=params['learning_rate'])
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=300)
        data_loader = self._getDataLoader(data, data.train_mask, batch_size=params['batch_size'])
        print("TRM: DataLoader created with " + str(len(data_loader)) + " batches")

        criterion = CrossEntropyLoss()
        best_val_loss = np.inf
        best_val_acc = np.inf
        best_train_loss = 0
        best_train_acc = 0
        worst_loss_times = 0
        for epoch in range(params['epochs']):

            # Train on batches
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

            if round(val_loss, 4) >= best_val_loss:
                worst_loss_times += 1
            else:
                worst_loss_times = 0
                best_val_loss = round(val_loss, 4)
                best_val_acc = round(val_acc, 4)
                best_train_loss = round(total_loss / len(data_loader), 4)
                best_train_acc = round(acc / len(data_loader), 4)
                # save best model's weights
                torch.save(self.model.state_dict(), 'tmp/temp_best_model_state_dict.pt')

            if worst_loss_times == params['earlyStoppingThresh']:
                break

        # reload best model's weights
        self.model.load_state_dict(torch.load('tmp/temp_best_model_state_dict.pt', map_location=torch.device(device)))

        if save_path is not None:
            with open(save_path, 'wb') as file:
                print("TRM: Saving model in pickle object")
                pickle.dump(self.model, file)
                print("TRM: Model saved")
                file.close()

        scores = {
            'train_loss': best_train_loss,
            'train_accuracy': best_train_acc,
            'validation_loss': best_val_loss,
            'validation_accuracy': best_val_acc,
            'epochs': epoch + 1
        }

        return scores

    def optimizeParameters(self, params):
        """
        Optimize train's parameters in params
        :param params:
        parameters used during train that will be optimized
        :return:
        A dictionary containing the loss of last train
        """

        global SavedParameters
        global best_loss

        start_train_time = np.datetime64(datetime.now())
        outs = self.train(params['input_graph'], params, params['device'])
        end_train_time = np.datetime64(datetime.now())

        torch.cuda.empty_cache()

        SavedParameters.append(outs)
        SavedParameters[-1].update({"learning_rate": params["learning_rate"],
                                    "train_time": str(end_train_time - start_train_time),
                                    "batch_size": params["batch_size"], "conv_layers_number": params["conv_layers"]})
        for i in range(params['conv_layers']):
            SavedParameters[-1].update({("dropout_" + str(i + 1)): params["dropout_" + str(i + 1)]})

        if SavedParameters[-1]["validation_loss"] < best_loss:
            if params['save_model_path'] is not None:
                print("OP: New saved model:" + str(SavedParameters[-1]))
                with open(params['save_model_path'], 'wb') as file:
                    print("OP: Saving model in pickle object")
                    pickle.dump(self.model, file)
                    print("OP: Model saved")
                    file.close()
            best_loss = SavedParameters[-1]["validation_loss"]

        SavedParameters = sorted(SavedParameters, key=lambda i: i['validation_loss'], reverse=False)

        with open(params['save_results_path'], 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=SavedParameters[0].keys())
            writer.writeheader()
            writer.writerows(SavedParameters)
            csvfile.close()

        scores = {
            'loss': outs['validation_loss'],
            'status': STATUS_OK
        }

        return scores

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


class GCN(Module):
    """
    Class to obtain a GCN model for homogeneous graphs
    """

    def __init__(self, num_node_features, num_classes, conv_layers_number, dropouts):
        super().__init__()
        torch.manual_seed(12)
        self.dropouts = dropouts

        channels = 512
        self.conv_layers = ModuleList([])
        self.conv_layers.append(GCNConv(num_node_features, channels))
        for i in range(conv_layers_number - 1):
            self.conv_layers.append(GCNConv(int(channels), int(channels / 2)))
            channels /= 2

        self.lin1 = Linear(int(channels), int(channels / 2), weight_initializer="glorot")
        self.lin2 = Linear(int(channels / 2), int(channels / 4), weight_initializer="glorot")
        self.lin3 = Linear(int(channels / 4), num_classes, weight_initializer="glorot")

    def forward(self, data, batch_node_indexes=None):
        x, edge_index, edge_weights = data.x, data.edge_index, data.edge_weight

        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x, edge_index, edge_weights)
            x = torch_functional.relu(x)
            x = torch_functional.dropout(x, p=self.dropouts[i])

        x = self.lin1(x)
        x = torch_functional.relu(x)
        x = self.lin2(x)
        x = torch_functional.relu(x)
        x = self.lin3(x)

        if batch_node_indexes is not None:
            x = x[batch_node_indexes]

        x = torch_functional.softmax(x, dim=1)
        return x
