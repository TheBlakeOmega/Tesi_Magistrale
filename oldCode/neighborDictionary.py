import pickle
import matplotlib.pyplot as plt
import numpy as np

"""
This class models a dictionary made by cosine similarity matrix. Each key-value pair in 'neighborDic' models a pair 
that contains the dataset example's index as key and the list of its neighbors as a list of examples' indexes.
An example j is a neighbor of the example i if the input cosine similarity matrix has a value greater than configuration
parameter 'minSimilarityValues' in the cell [i,j].
Each key-value pair in 'neighborCountDic' models a pair that contains the dataset example's index as key and how many
neighbors it has

Init Parameters
----------
:param configuration:
program configuration
:param data_type:
string that specifies if data belongs to train or test set
"""


class NeighborDictionary:

    def __init__(self, configuration, data_type):
        self.neighborDic = dict()
        self.neighborCountDic = dict()
        self.configuration = configuration
        self.data_type = data_type

    """
        This method computes neighborDic and neighborCountDic contained in the class
        Parameters
        ----------
        :param matrix:
        cosine similarity matrix
        :param serialize:
        boolean that specifies if neighborDic and neighborCountDic will be serialized in pickle objects
    """

    def buildDictionary(self, matrix, serialize=True):
        print("BND: Computing " + self.data_type + " neighborhood dictionary")
        for rowIndex, row in enumerate(matrix):
            self.neighborDic[rowIndex] = []
            self.neighborCountDic[rowIndex] = 0
            for columnIndex, value in enumerate(row):
                if rowIndex != columnIndex and (value * 100) >= float(self.configuration['minSimilarityValues']):
                    self.neighborDic[rowIndex].append(columnIndex)
                    self.neighborCountDic[rowIndex] += 1

            if rowIndex % 100 == 0:
                print("BND: " + str(rowIndex) + " examples computed")

        if serialize:
            with open(self.configuration['pathSimilarityDictionary'] + self.configuration['minSimilarityValues'] +
                      "_similarity/" + self.configuration['chosenDataset'] + "_" + self.data_type +
                      "_similarity_dictionary.pkl", 'wb') as file:
                print("BND: Saving " + self.data_type + " similarity dictionary in pickle object")
                pickle.dump(self.neighborDic, file)
                print("BND: Dictionary saved")
                file.close()
            with open(self.configuration['pathSimilarityDictionary'] + self.configuration['minSimilarityValues'] +
                      "_similarity/" + self.configuration['chosenDataset'] + "_" + self.data_type +
                      "_similarity_dictionary_count.pkl", 'wb') as file:
                print("BND: Saving " + self.data_type + " similarity dictionary count in pickle object")
                pickle.dump(self.neighborCountDic, file)
                print("BND: Count dictionary saved")
                file.close()

    """
       This method loads neighborDic and neighborCountDic from storage
    """

    def loadDictionary(self):
        with open(self.configuration['pathSimilarityDictionary'] + self.configuration['minSimilarityValues'] +
                  "_similarity/" + self.configuration['chosenDataset'] + "_" + self.data_type +
                  "_similarity_dictionary.pkl", 'rb') as file:
            print("LND: Loading " + self.data_type + " similarity dictionary from " +
                  self.configuration['pathSimilarityDictionary'] + self.configuration['minSimilarityValues'] +
                  "_similarity/" + self.configuration['chosenDataset'] + "_" + self.data_type +
                  "_similarity_dictionary.pkl")
            self.neighborDic = pickle.load(file)
            print("LND: Dictionary loaded")
            file.close()
        with open(self.configuration['pathSimilarityDictionary'] + self.configuration['minSimilarityValues'] +
                  "_similarity/" + self.configuration['chosenDataset'] + "_" + self.data_type +
                  "_similarity_dictionary_count.pkl", 'rb') as file:
            print("LND: Loading " + self.data_type + " similarity dictionary count from " +
                  self.configuration['pathSimilarityDictionary'] + self.configuration['minSimilarityValues'] +
                  "_similarity/" + self.configuration['chosenDataset'] + "_" + self.data_type +
                  "_similarity_dictionary_count.pkl")
            self.neighborCountDic = pickle.load(file)
            print("LND: Dictionary count loaded")
            file.close()

    """
        This method counts how many examples has no neighbors
        Returns
        ----------
        :count:
        number of examples that has no neighbors
    """

    def countIsolatedExamples(self):
        count = 0
        for neighborCount in list(self.neighborCountDic.values()):
            if neighborCount == 0:
                count += 1
        return count

    """
        This method builds and serialize a boxplot that shows example number of neighbors' density
    """

    def buildNeighborCountBoxPlot(self):
        print("BPND: Building boxplot on " + self.data_type + " neighbor count")
        plt.figure(figsize=(10, 7))
        plt.boxplot(list(self.neighborCountDic.values()))
        print("BPND: Saving boxplot on " + self.data_type + " neighbor count")
        plt.savefig(
            "results/plots/boxplot/" + self.configuration['chosenDataset'] + "_" + self.data_type +
            "_neighbor_count_boxplot_" + self.configuration['minSimilarityValues'] + ".jpeg")
        print("BPND: Boxplot saved\n")
        plt.close()

    """
            This method computes evaluation metrics on count dictionary
            Returns
            ----------
            :max_value:
            number of neighbors from the examples that has maximum number of them
            :min_value:
            number of neighbors from the examples that has minimum number of them
            :mean_value:
            mean number of neighbors 
            :standard_deviation:
            standard deviation on neighbors count
    """

    def computeMesuresOnNeighborsCount(self):
        max_value = np.max(list(self.neighborCountDic.values()))
        min_value = np.min(list(self.neighborCountDic.values()))
        mean_value = np.mean(list(self.neighborCountDic.values()))
        standard_deviation = np.std(list(self.neighborCountDic.values()))
        return max_value, min_value, mean_value, standard_deviation
