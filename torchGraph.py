import torch
import torch_directml
import pandas as pd
import pickle
from torch_geometric.data import Data

"""
This class models a graph made by torch_geometric 

Init Parameters
----------
:param configuration:
program configuration
:param dataset_configuration:
dataset configuration
:param data_type:
string that specifies if data belongs to train or test set
"""


class TorchGraph:

    def __init__(self, configuration, dataset_configuration, data_type):
        self.configuration = configuration
        self.ds_configuration = dataset_configuration
        self.device = torch_directml.device()
        self.data_type = data_type
        self.graph = None

    """
        This method builds the graph computing nodes from feature_data and edges from neighborhood relationship
        specified in neighbor_dictionary. Edges weights are taken from similarity_matrix.
        Parameters
        ----------
        :param feature_data:
        examples from dataset that will be nodes' attributes
        :param label_data:
        example's label that will be nodes' target attribute 
        :param neighbor_dictionary:
        dictionary with <int - List<int>> pairs that specifies neighborhood relationship between nodes to build edges
        :param cosine_similarity_matrix:
        similarity matrix, whose values will be edges' weight
        :param serialize:
        boolean to specify if serialize the graph built
    """

    def buildGraph(self, feature_data, label_data, neighbor_dictionary, cosine_similarity_matrix, serialize=True):

        print("BTG: Creating torch geometric tensors from data for dataset " + self.configuration['chosenDataset'])
        features = torch.tensor(feature_data.values, dtype=torch.float).to(self.device)
        labels = torch.tensor(label_data.values, dtype=torch.long).to(self.device)

        print("BTG: Creating edge dataframe for dataset " + self.configuration['chosenDataset'])
        edge_dataframe = pd.DataFrame(columns=['source', 'label', 'weight'])
        for example_index in neighbor_dictionary:
            greater_neighbor_indexes = [i for i in neighbor_dictionary[example_index] if i > example_index]
            for neighbor_index in greater_neighbor_indexes:
                edge_dataframe.loc[len(edge_dataframe)] = [example_index, neighbor_index,
                                                           cosine_similarity_matrix[example_index][neighbor_index]]
            if example_index % 100 == 0:
                print("BTG: " + str(example_index) + " edges computed")

        print("BTG: Creating torch geometric tensors from edges for dataset " + self.configuration['chosenDataset'])
        edge_index = torch.tensor(
            [edge_dataframe['source'], edge_dataframe['label']],
            dtype=torch.long
        ).to(self.device)
        edge_weights = torch.tensor(edge_dataframe['weight'], dtype=torch.float).to(self.device)

        print("BTG: Creating torch geometric graph for dataset " + self.configuration['chosenDataset'])
        self.graph = Data(x=features, y=labels, edge_index=edge_index, edge_weight=edge_weights,
                          num_classes=label_data.nunique())

        print("Graph info:", self.graph)

        if serialize:
            with open(self.configuration['pathPytorchGraphs'] + self.configuration['chosenDataset'] + "_" +
                      self.data_type + "pytorch_graph.pkl", 'wb') as file:
                print("Saving " + self.data_type + " similarity dictionary in pickle object")
                pickle.dump(self.graph, file)
                print("Dictionary saved")
                file.close()

    """
        This method loads a graph from storage
    """

    def loadGraph(self):
        with open(self.configuration['pathPytorchGraphs'] + self.configuration['chosenDataset'] + "_" + self.data_type +
                  "pytorch_graph.pkl", 'rb') as file:
            print("LTG: Loading " + self.data_type + " Pytorch Graph from " + self.configuration['pathPytorchGraphs'] +
                  self.configuration['chosenDataset'] + "_" + self.data_type + "pytorch_graph.pkl")
            self.graph = pickle.load(file)
            print("LTG: Pytorch Graph loaded")
            file.close()



