from dataset import Dataset
from cosineSimilarityMatrix import CosineSimilarityMatrix
import pca
from neighborDictionary import NeighborDictionary
import traceback

"""
This class is a manager for the pipelines that user wants to run. Method 'runPipeline' runs the pipeline chosen 
according to the configuration file

Init Parameters
----------
:param configuration:
program configuration
:param dataset_configuration:
dataset configuration
:param chosen_pipeline:
string with pipeline chosen in the configuration file
:param result_file:
txt file to write results
"""


class PipeLineManager:

    def __init__(self, configuration, dataset_configuration, chosen_pipeline, result_file):
        self.configuration = configuration
        self.ds_configuration = dataset_configuration
        self.chosen_pipeline = chosen_pipeline
        self.result_file = result_file
        print("\n LOADING DATASET \n")
        self.train_dataset = Dataset(dataset_configuration)
        self.test_dataset = Dataset(dataset_configuration)
        self.train_dataset.loadDataset(dataset_type='Train')
        self.test_dataset.loadDataset(dataset_type='Test')

    """
    This method runs the pipeline chosen according to the configuration file
    """

    def runPipeline(self):
        if self.chosen_pipeline == 'SIMILARITY_MATRIX':
            try:
                self._buildCosineSimilarityMatrix()
                print("Similarity matrices built successfully.")
                self.result_file.write("Similarity matrices built successfully.")
            except Exception as e:
                traceback.print_exc()
                raise Exception("Error during Neighbor dictionaries creation")
        elif self.chosen_pipeline == 'NEIGHBOR_DICTIONARY':
            try:
                self._buildNeighborDictionary()
                print("Neighbor dictionaries built successfully.")
                self.result_file.write("Neighbor dictionaries built successfully.")
            except Exception as e:
                traceback.print_exc()
                raise Exception("Error during Neighbor dictionaries creation")

    """
        This method runs the pipeline to build and serialize similarity matrices
    """

    def _buildCosineSimilarityMatrix(self):
        print("\n GETTING COSINE SIMILARITY MATRICES \n")
        train_matrix = CosineSimilarityMatrix(self.configuration, data_type="train")
        test_matrix = CosineSimilarityMatrix(self.configuration, data_type="test")
        print("Building PCA model\n")
        pca_model = pca.buildPCAModel(self.train_dataset.features, self.configuration['chosenDataset']
                                      , float(self.configuration['PCA_n_components']))
        print("Applying PCA model on data\n")
        train_PCA_components = pca.applyPCA(self.train_dataset.features, pca_model)
        test_PCA_components = pca.applyPCA(self.test_dataset.features, pca_model)
        train_matrix.buildMatrix(train_PCA_components)
        test_matrix.buildMatrix(train_PCA_components, test_PCA_components)
        print("\n")
        self.result_file.write("Train similarity matrix:\n" + str(train_matrix.matrix) + " \nShapes: " +
                               str(train_matrix.matrix.shape) + "\n\n")
        self.result_file.write("Test similarity matrix:\n" + str(test_matrix.matrix) + " \nShapes: " +
                               str(test_matrix.matrix.shape) + "\n\n")

    """
        This method runs the pipeline to build and serialize neighbor dictionaries
    """

    def _buildNeighborDictionary(self):
        print("\n GETTING NEIGHBORHOOD DICTIONARIES \n")

        train_matrix = CosineSimilarityMatrix(self.configuration, data_type="train")
        test_matrix = CosineSimilarityMatrix(self.configuration, data_type="test")
        train_matrix.loadMatrix()
        test_matrix.loadMatrix()

        train_dictionary = NeighborDictionary(self.configuration, data_type="train")
        test_dictionary = NeighborDictionary(self.configuration, data_type="test")
        train_dictionary.buildDictionary(train_matrix.matrix)
        test_dictionary.buildDictionary(test_matrix.matrix)
        train_dictionary.buildNeighborCountBoxPlot()
        self.result_file.write("Number of isolated nodes in train matrix: " +
                               str(train_dictionary.countIsolatedExamples()) + "\n\n")
        test_dictionary.buildNeighborCountBoxPlot()
        self.result_file.write("Number of isolated nodes in test matrix: " +
                               str(test_dictionary.countIsolatedExamples()) + "\n\n")
