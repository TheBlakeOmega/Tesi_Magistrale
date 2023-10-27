from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
import pandas as pd


class CosineSimilarityMatrix:
    """
    This class models a matrix where each cell with indexes i and j contains cosine similarity between example with
    index i in rows_data and example with index j in columns_data.
    """

    def __init__(self):
        self.matrix = None
        self.rowsEqualToColumns = False

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
              (str(columns.shape[0]) if columns is not None else str(rows.shape[0])))
        self.rowsEqualToColumns = columns is None
        matrix_float64 = cosine_similarity(rows, columns if columns is not None else rows)
        print("BSM: Reducing similarity matrix values' size")
        self.matrix = np.array(matrix_float64, dtype=np.float16)
        if save_path is not None:
            with open(save_path, 'wb') as file:
                print("BSM: Saving similarity matrix in pickle object")
                pickle.dump(self.matrix, file)
                print("BSM: Matrix saved")
                file.close()

    def loadMatrix(self, load_path):
        """
        This method loads a matrix from storage
        :param load_path:
        path from which load the matrix
        """
        with open(load_path, 'rb') as file:
            print("LSM: Loading similarity matrix from " + load_path)
            self.matrix = pickle.load(file)
            self.rowsEqualToColumns = self.matrix.shape[0] == self.matrix.shape[1]
            print("LSM: Matrix loaded")
            file.close()

    def getMatrix(self):
        """
        This method returns class's matrix attribute
        :return:
        similarity matrix
        """
        return self.matrix

    def areRowsEqualToColumns(self):
        """
        This method returns True if data used as rows were equals to Columns, False otherwise
        :return:
        rowsEqualToColumns attribute
        """
        return self.rowsEqualToColumns

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

    def generateTorchGraph(self, dataset_row, threshold, device, dataset_column=None, path=None):
        """
        This method generate a pytorch graph modeled by torch_geometric.data.Data class
        :param dataset_row:
        Dataset with data corresponding to similarity matrix's rows
        :param threshold:
        threshold value from which a node can be considered a neighbor
        :param device:
        device that will compute tensors
        :param dataset_column:
        default None. If specified, Dataset with data corresponding to similarity matrix's columns
        :param path:
        default None. If specified, it will be the path in which the graph created will be saved
        :return:
        an instance of torch_geometric.data.Data with built graph
        """
        feature_data = dataset_row.getFeatureData()
        edge_list = []
        if self.rowsEqualToColumns:
            print("GTG: Creating edge dataframe")
            for rowIndex, row in enumerate(self.matrix):
                for columnIndex, value in enumerate(row):
                    if rowIndex > columnIndex:
                        if value * 100 >= threshold:
                            edge_list.append([rowIndex, columnIndex, self.matrix[rowIndex][columnIndex]])
                    else:
                        break
                if rowIndex % 100 == 0:
                    print("GTG: " + str(rowIndex) + " X examples computed")

        else:
            rows_number = len(feature_data.index)
            feature_data = pd.concat([feature_data, dataset_column.getFeatureData()], ignore_index=True)
            print("GTG: Creating edge dataframe")
            for rowIndex, row in enumerate(self.matrix):
                for columnIndex, value in enumerate(row):
                    if rowIndex != columnIndex and value * 100 >= threshold:
                        edge_list.append([rowIndex, rows_number + columnIndex, self.matrix[rowIndex][columnIndex]])
                if rowIndex % 100 == 0:
                    print("GTG: " + str(rowIndex) + " X examples computed")

        print("GTG: Creating torch geometric tensors from data")
        features = torch.tensor(feature_data.values, dtype=torch.float).to(device)
        labels = torch.tensor(dataset_row.getLabelData().values, dtype=torch.long).to(device)
        edge_dataframe = pd.DataFrame(edge_list, columns=['source', 'target', 'weight'])
        edge_index = torch.tensor([edge_dataframe['source'], edge_dataframe['target']], dtype=torch.long).to(device)
        edge_weights = torch.tensor(edge_dataframe['weight'], dtype=torch.float).to(device)

        print("BTG: Creating torch geometric graph from built tensors")
        graph = Data(x=features, y=labels, edge_index=edge_index, edge_weight=edge_weights,
                     num_classes=dataset_row.getLabelData().nunique()[0]).to(device)

        print("Graph info:", graph)

        if path is not None:
            with open(path, 'wb') as file:
                print("BTG: Saving graph in pickle object at " + path)
                pickle.dump(graph, file)
                print("BTG: Graph saved")
                file.close()

        return graph
