from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import matplotlib as plt
import torch
from torch_geometric.data import Data
import pandas as pd
from abc import ABC, abstractmethod
import torch_geometric.transforms as torch_transform
import operator


class CosineSimilarityMatrix(ABC):
    """
    This class models a matrix where each cell with indexes i and j contains cosine similarity between example with
    index i in rows_data and example with index j in columns_data.
    """

    def __init__(self):
        self.matrix = None

    @abstractmethod
    def computeMatrix(self, rows, columns=None, save_path=None):
        pass

    def loadMatrix(self, load_path):
        """
        This method loads a matrix from storage
        :param load_path:
        path from which load the matrix
        """
        with open(load_path, 'rb') as file:
            print("LSM: Loading similarity matrix from " + load_path)
            self.matrix = pickle.load(file)
            print("LSM: Matrix loaded")
            file.close()

    def getMatrix(self):
        """
        This method returns class's matrix attribute
        :return:
        similarity matrix
        """
        return self.matrix

    def buildSimilarityBoxPlot(self, path=None):
        """
        This method builds a boxplot that shows matrix values' density
        :param path:
        default None. If specified, it will be the path in which the boxplot created will be saved
        """
        print("BPSM: Building boxplot on matrix's values")
        plt.figure(figsize=(10, 7))
        plt.boxplot(self.matrix.flatten())
        plt.show()
        if path is not None:
            print("BPSM: Saving boxplot on matrix's values")
            plt.savefig(path)
            print("BPSM: Boxplot saved\n")
        plt.close()

    def buildNeighborCountBoxPlotAndReport(self, threshold, path=None):
        """
        This method builds a boxplot that shows example number of neighbors' density and computes report metrics
        :param threshold:
        threshold value from which a node can be considered a neighbor
        :param path:
        default None. If specified, it will be the path in which the boxplot created will be saved
        :return:
        metrics on neighbor count (max_value, min_value, mean_value, standard_deviation, number_isolated_examples)
        """
        print("BPNC: Computing neighborhood dictionary")
        count_neighbor_list = []
        number_isolated_examples = 0
        for rowIndex, row in enumerate(self.matrix):
            neighbor_count = 0
            for columnIndex, value in enumerate(row):
                if rowIndex != columnIndex and value * 100 >= threshold:
                    neighbor_count += 1
            if neighbor_count == 0:
                number_isolated_examples += 1
            count_neighbor_list.append(neighbor_count)

            if rowIndex % 100 == 0:
                print("BPNC: " + str(rowIndex) + " examples computed")

        print("BPNC: Building boxplot on neighbor count")
        plt.figure(figsize=(10, 7))
        plt.boxplot(count_neighbor_list)
        plt.show()
        if path is not None:
            print("BPNC: Saving boxplot on neighbor count")
            plt.savefig(path)
            print("BPNC: Boxplot saved\n")
        plt.close()

        max_value = np.max(count_neighbor_list)
        min_value = np.min(count_neighbor_list)
        mean_value = np.mean(count_neighbor_list)
        standard_deviation = np.std(count_neighbor_list)
        return max_value, min_value, mean_value, standard_deviation, number_isolated_examples


class CosineSimilarityMatrixTrain(CosineSimilarityMatrix):
    """
    Train cosine similarity matrix
    """
    def computeMatrix(self, rows, columns=None, save_path=None):
        """
        This method compute the matrix contained in the class
        Parameters
        ----------
        :param rows:
        dataSet that will be used as example on matrix's rows
        :param columns:
        dataSet that will be used as example on matrix's columns. If this parameter is None, rows_data parameter will be
        used
        :param save_path:
        boolean that specifies fi the matrix will be serialized in a pickle object
        """
        print("BSM: Computing similarity matrix with shape: " + str(rows.shape[0]) + "," +
              str(rows.shape[0]))
        matrix_float64 = cosine_similarity(rows, rows)
        print("BSM: Reducing similarity matrix values' size")
        self.matrix = np.array(matrix_float64, dtype=np.float16)
        if save_path is not None:
            with open(save_path, 'wb') as file:
                print("BSM: Saving similarity matrix in pickle object")
                pickle.dump(self.matrix, file)
                print("BSM: Matrix saved")
                file.close()

    def generateTrainTorchGraph(self, dataset_row, threshold, max_neighbors, device, path=None):
        """
        This method generate a pytorch graph modeled by torch_geometric.data.Data class
        :param dataset_row:
        Dataset with data corresponding to similarity matrix's rows
        :param threshold:
        threshold value from which a node can be considered a neighbor
        :param max_neighbors:
        max number of neighbour that each node can have
        :param device:
        device that will compute tensors
        :param path:
        default None. If specified, it will be the path in which the graph created will be saved
        :return:
        an instance of torch_geometric.data.Data with built graph
        """
        feature_data = dataset_row.getFeatureData()
        edge_list = []
        print("GTG: Creating edge dataframe")
        for rowIndex, row in enumerate(self.matrix):
            indexed_row = list(enumerate(row))
            top_similarities = sorted(indexed_row, key=operator.itemgetter(1))[-(max_neighbors + 1):]
            for (columnIndex, similarity) in top_similarities:
                if rowIndex != columnIndex and similarity * 100 >= threshold:
                    edge_list.append([rowIndex, columnIndex, similarity])
            if rowIndex % 100 == 0:
                print("GTG: " + str(rowIndex) + " X examples computed")
        print("GTG: Creating train torch geometric tensors from data")
        features = torch.tensor(feature_data.values, dtype=torch.float).to(device)
        labels = torch.tensor(dataset_row.getLabelData().values, dtype=torch.long).to(device)
        edge_dataframe = pd.DataFrame(edge_list, columns=['source', 'target', 'weight'])
        edge_index = torch.tensor([edge_dataframe['source'], edge_dataframe['target']], dtype=torch.long).to(device)
        edge_weights = torch.tensor(edge_dataframe['weight'], dtype=torch.float).to(device)
        torch_train_mask, torch_validation_mask = dataset_row.computeMasks()
        torch_train_mask = torch_train_mask.to(device)
        torch_validation_mask = torch_validation_mask.to(device)
        print("BTG: Creating train torch geometric graph from built tensors")
        graph = Data(x=features, y=labels, edge_index=edge_index, edge_weight=edge_weights,
                     num_classes=dataset_row.getLabelData().nunique().iloc[0], num_nodes=len(feature_data)).to(device)
        graph = graph.coalesce()
        graph.train_mask = torch_train_mask
        graph.validation_mask = torch_validation_mask
        print("Train Graph info:", graph)

        if path is not None:
            with open(path, 'wb') as file:
                print("BTG: Saving train graph in pickle object at " + path)
                pickle.dump(graph, file)
                print("BTG: Train graph saved")
                file.close()
        return graph


class CosineSimilarityMatrixTest(CosineSimilarityMatrix):
    """
    Test cosine similarity matrix
    """

    def computeMatrix(self, rows, columns=None, save_path=None):
        """
        This method compute the matrix contained in the class
        Parameters
        ----------
        :param rows:
        dataSet that will be used as example on matrix's rows
        :param columns:
        dataSet that will be used as example on matrix's columns. If this parameter is None, rows_data parameter will be
        used
        :param save_path:
        boolean that specifies fi the matrix will be serialized in a pickle object
        """
        if columns is None:
            raise Exception("In test cosine similarity matrix columns dataframe must not be None")
        print("BSM: Computing test similarity matrix with shape: " + str(rows.shape[0]) + "," +
              str(columns.shape[0]))
        matrix_float64 = cosine_similarity(rows, columns)
        print("BSM: Reducing test similarity matrix values' size")
        self.matrix = np.array(matrix_float64, dtype=np.float16)
        if save_path is not None:
            with open(save_path, 'wb') as file:
                print("BSM: Saving test similarity matrix in pickle object")
                pickle.dump(self.matrix, file)
                print("BSM: Test matrix saved")
                file.close()

    def addTestExampleToGraph(self, threshold, train_graph, test_example, test_example_index, device, path=None):
        """
        This method add a new test example to an existent input torch_geometric graph
        :param threshold:
        threshold value from which a node can be considered a neighbor
        :param train_graph:
        torch_geometric graph with train data
        :param test_example:
        test example that will be added to the train graph
        :param test_example_index:
        index of the test example
        :param device:
        device that will compute tensors
        :param path:
        default None. If specified, it will be the path in which the graph created will be saved
        :return:
        a copy of the train graph with a new test example
        """
        x = train_graph.x.tolist()
        edge_weights = train_graph.edge_weight.tolist()
        edge_index = train_graph.edge_index.tolist()
        new_example_index = len(x)
        x.append(test_example)
        for rowIndex, row in enumerate(self.matrix):
            similarity = self.matrix[rowIndex][test_example_index]
            if similarity * 100 >= threshold:
                edge_weights.extend([similarity, similarity])
                edge_index[0].extend([rowIndex, new_example_index])
                edge_index[1].extend([new_example_index, rowIndex])
        x = torch.tensor(x, dtype=torch.float).to(device)
        edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)
        edge_weights = torch.tensor(edge_weights, dtype=torch.float).to(device)

        test_graph = Data(x=x, y=train_graph.y, edge_index=edge_index, edge_weight=edge_weights,
                          num_classes=train_graph.num_classes, num_nodes=(train_graph.num_nodes + 1)).to(device)

        if path is not None:
            with open(path, 'wb') as file:
                print("BTG: Saving test graph in pickle object at " + path)
                pickle.dump(test_graph, file)
                print("BTG: Test graph saved")
                file.close()

        return test_graph
