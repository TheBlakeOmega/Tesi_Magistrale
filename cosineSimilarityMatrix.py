from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pickle
import numpy as np

"""
This class models a matrix where each cell with indexes i and j contains cosine similarity between example with index i
in rows_data and example with index j in columns_data.

Init Parameters
----------
:param configuration:
program configuration
:param data_type:
string that specifies if data belongs to train or test set
"""


class CosineSimilarityMatrix:

    def __init__(self, configuration, data_type):
        self.configuration = configuration
        self.matrix = None
        self.data_type = data_type

    """
    This method compute the matrix contained in the class
    Parameters
    ----------
    :param rows_data:
    dataSet that will be used as example on matrix's rows
    :param columns_data:
    dataSet that will be used as example on matrix's columns. If this parameter is None, rows_data parameter will be 
    used
    :param serialize:
    boolean that specifies fi the matrix will be serialized in a pickle object
    """

    def buildMatrix(self, rows_data, columns_data=None, serialize=True):
        print("BSM: Computing " + self.data_type + " similarity matrix with shape: " + str(rows_data.shape[0]) + "," +
              (str(columns_data.shape[0]) if columns_data is not None else str(rows_data.shape[0])))
        matrix_float64 = cosine_similarity(rows_data, columns_data if columns_data is not None else rows_data)
        print("BSM: Reducing " + self.data_type + " similarity matrix values' size")
        self.matrix = np.array(matrix_float64, dtype=np.float16)
        if serialize:
            with open(self.configuration['pathSimilarityMatrices'] + self.configuration['chosenDataset'] +
                      "_" + self.data_type + "_similarity_matrix.pkl", 'wb') as file:
                print("BSM: Saving " + self.data_type + " similarity matrix in pickle object")
                pickle.dump(self.matrix, file)
                print("BSM: Matrix saved")
                file.close()

    """
        This method build and serialize a boxplot that shows matrix values' density
    """

    def buildValuesBoxPlot(self):
        print("BPSM: Building boxplot on " + self.data_type + " matrix's values")
        plt.figure(figsize=(10, 7))
        plt.boxplot(self.matrix.flatten())
        print("BPSM: Saving boxplot on " + self.data_type + " matrix's values")
        plt.savefig(
            "results/plots/boxplot/" + self.configuration['chosenDataset'] + "_" + self.data_type +
            "_similarity_matrix_boxplot.jpeg")
        print("BPSM: Boxplot saved\n")
        plt.close()

    """
       This method loads a matrix from storage
    """

    def loadMatrix(self):
        with open(self.configuration['pathSimilarityMatrices'] + self.configuration['chosenDataset'] +
                  "_" + self.data_type + "_similarity_matrix.pkl", 'rb') as file:
            print("LSM: Loading " + self.data_type + " similarity matrix from " +
                  self.configuration['pathSimilarityMatrices'] + self.configuration['chosenDataset'] +
                  "_" + self.data_type + "train_similarity_matrix.pkl")
            self.matrix = pickle.load(file)
            print("LSM: Matrix loaded")
            file.close()
