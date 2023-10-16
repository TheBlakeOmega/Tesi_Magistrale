import configparser
import numpy as np
import pca
import printFunctions

from cosineSimilarityMatrix import CosineSimilarityMatrix
from dataset import Dataset
from neighborDictionary import NeighborDictionary

np.random.seed(12)


def datasetException(dataset_name):
    try:
        if dataset_name is None:
            raise TypeError()
        if not ((dataset_name == 'MALDROID') or (dataset_name == 'MALMEM')
                or (dataset_name == 'NSL_KDD') or (dataset_name == 'UNSWNUMERIC')):
            raise ValueError()
        return dataset_name
    except ValueError:
        print("Dataset not exist: must be MALDROID or MALMEM or NSL_KDD")
    except TypeError:
        print("The name of dataset is null: use MALDROID or MALMEM or NSL_KDD")


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('configuration.conf')
    configuration = config['SETTINGS']
    dataset = datasetException(configuration['chosenDataset'])
    dsConf = config[dataset]

    with open("results/result_" + dataset + "_" + configuration['minSimilarityValues'] + ".txt", "w") as result_file:

        print("\n LOADING DATASET \n")
        train_dataset = Dataset(dsConf)
        test_dataset = Dataset(dsConf)
        train_dataset.loadDataset(dataset_type='Train')
        test_dataset.loadDataset(dataset_type='Test')

        print("\n GETTING COSINE SIMILARITY MATRICES \n")
        train_matrix = CosineSimilarityMatrix(configuration, data_type="train")
        test_matrix = CosineSimilarityMatrix(configuration, data_type="test")
        try:
            train_matrix.loadMatrix()
            test_matrix.loadMatrix()
        except OSError as e:
            print("Building PCA model\n")
            pcaModel = pca.buildPCAModel(train_dataset.features, configuration['chosenDataset']
                                         , float(configuration['PCA_n_components']))
            print("Applying PCA model on data\n")
            train_PCA_components = pca.applyPCA(train_dataset.features, pcaModel)
            test_PCA_components = pca.applyPCA(test_dataset.features, pcaModel)
            train_matrix.buildMatrix(train_PCA_components)
            test_matrix.buildMatrix(train_PCA_components, test_PCA_components)
            print("\n")

        result_file.write("Train similarity matrix:\n" + str(train_matrix.matrix) + " \nShapes: " +
                          str(train_matrix.matrix.shape) + "\n\n")
        result_file.write("Test similarity matrix:\n" + str(test_matrix.matrix) + " \nShapes: " +
                          str(test_matrix.matrix.shape) + "\n\n")

        print("\n GETTING NEIGHBORHOOD DICTIONARIES \n")
        train_dictionary = NeighborDictionary(configuration, data_type="train")
        test_Dictionary = NeighborDictionary(configuration, data_type="test")
        try:
            train_dictionary.loadDictionary()
        except OSError as e:
            train_dictionary.buildDictionary(train_matrix.matrix)
        train_dictionary.buildNeighborCountBoxPlot()
        result_file.write("Number of isolated nodes in train matrix: " + str(train_dictionary.countIsolatedExamples())
                          + "\n\n")
        printFunctions.printDictionaryOnFile("Train neighborhood count dictionary with percentage at " +
                                             configuration['minSimilarityValues'] + ":",
                                             train_dictionary.neighborCountDic, result_file)

        result_file.close()
