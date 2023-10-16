import pickle
import matplotlib.pyplot as plt

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
        This method compute neighborDic and neighborCountDic contained in the class
        Parameters
        ----------
        :param matrix:
        cosine similarity matrix
        :param serialize:
        boolean that specifies if neighborDic and neighborCountDic will be serialized in pickle objects
    """

    def buildDictionary(self, matrix, serialize=True):
        print("Computing " + self.data_type + " neighborhood dictionary")
        for rowIndex, row in enumerate(matrix):
            self.neighborDic[rowIndex] = []
            self.neighborCountDic[rowIndex] = 0
            for columnIndex, value in enumerate(row):
                if rowIndex != columnIndex and (value * 100) >= float(self.configuration['minSimilarityValues']):
                    self.neighborDic[rowIndex].append(columnIndex)
                    self.neighborCountDic[rowIndex] += 1

            if rowIndex % 100 == 0:
                print("ND: " + str(rowIndex) + " examples computed")

        if serialize:
            with open(self.configuration['pathSimilarityDictionary'] + self.configuration['chosenDataset'] +
                      "_" + self.data_type + "_similarity_dictionary.pkl", 'wb') as file:
                print("Saving " + self.data_type + " similarity dictionary in pickle object")
                pickle.dump(self.neighborDic, file)
                print("Dictionary saved")
                file.close()
            with open(self.configuration['pathSimilarityDictionary'] + self.configuration['chosenDataset'] +
                      "_" + self.data_type + "_similarity_dictionary_count.pkl", 'wb') as file:
                print("Saving " + self.data_type + " similarity dictionary count in pickle object")
                pickle.dump(self.neighborCountDic, file)
                print("Count dictionary saved")
                file.close()

    """
       This method loads neighborDic and neighborCountDic from storage
    """

    def loadDictionary(self):
        with open(self.configuration['pathSimilarityDictionary'] + self.configuration['chosenDataset'] +
                  "_" + self.data_type + "_similarity_dictionary.pkl", 'rb') as file:
            print("Loading " + self.data_type + " similarity dictionary from " +
                  self.configuration['pathSimilarityDictionary'] + self.configuration['chosenDataset'] +
                  "_" + self.data_type + "_similarity_dictionary.pkl")
            self.neighborDic = pickle.load(file)
            print("Dictionary loaded")
            file.close()
        with open(self.configuration['pathSimilarityDictionary'] + self.configuration['chosenDataset'] +
                  "_" + self.data_type + "_similarity_dictionary_count.pkl", 'rb') as file:
            print("Loading " + self.data_type + " similarity dictionary count from " +
                  self.configuration['pathSimilarityDictionary'] + self.configuration['chosenDataset'] +
                  "_" + self.data_type + "_similarity_dictionary_count.pkl")
            self.neighborCountDic = pickle.load(file)
            print("Dictionary count loaded")
            file.close()

    """
        This method count how many examples has no neighbors
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
        This method build and serialize a boxplot that shows example number of neighbors' density
    """

    def buildNeighborCountBoxPlot(self):
        print("Building boxplot on " + self.data_type + " neighbor count")
        plt.figure(figsize=(10, 7))
        plt.boxplot(list(self.neighborCountDic.values()))
        print("Saving boxplot on " + self.data_type + " neighbor count")
        plt.savefig(
            "results/plots/boxplot/" + self.configuration['chosenDataset'] + "_" + self.data_type +
            "_neighbor_count_boxplot_" + self.configuration['minSimilarityValues'] + ".jpeg")
        print("Boxplot saved\n")
        plt.close()
