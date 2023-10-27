from dataset import Dataset
from cosineSimilarityMatrix import CosineSimilarityMatrix
import traceback
import torch_directml


class PipeLineManager:
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

    def __init__(self, configuration, dataset_configuration, chosen_pipeline, result_file):
        self.configuration = configuration
        self.ds_configuration = dataset_configuration
        self.chosen_pipeline = chosen_pipeline
        self.result_file = result_file

    def runPipeline(self):
        """
        This method runs the pipeline chosen according to the configuration file
        """
        if self.chosen_pipeline == 'BUILD_GRAPH':
            try:
                self._buildGraph()
                print("Similarity matrices built successfully.")
                self.result_file.write("Similarity matrices built successfully.")
            except Exception as e:
                traceback.print_exc()
                raise Exception("Error during Graph creation")

    def _buildGraph(self):
        """
        This method runs the pipeline to build and serialize similarity matrices and graph
        """
        print("\n LOADING DATASET \n")
        train_dataset = Dataset(self.ds_configuration['pathTrainDataset'],
                                self.ds_configuration['labelColumnName'])
        print("LDS: Train data loaded:\n    columns:" + str(len(train_dataset.getFeatureData().columns))
              + '    rows:' + str(len(train_dataset.getFeatureData().index)) + "\n\n")
        test_dataset = Dataset(self.ds_configuration['pathTestDataset'],
                               self.ds_configuration['labelColumnName'])
        print("LDS: Test data loaded:\n    columns:" + str(len(test_dataset.getFeatureData().columns))
              + '    rows:' + str(len(test_dataset.getFeatureData().index)) + "\n\n")

        print("\n GETTING COSINE SIMILARITY MATRICES \n")
        train_matrix = CosineSimilarityMatrix()
        test_matrix = CosineSimilarityMatrix()
        try:
            train_matrix.loadMatrix(self.configuration['pathSimilarityMatrices'] + self.configuration['chosenDataset'] +
                                    "_train_similarity_matrix.pkl")
            test_matrix.loadMatrix(self.configuration['pathSimilarityMatrices'] + self.configuration['chosenDataset'] +
                                   "_test_similarity_matrix.pkl")
        except FileNotFoundError:
            print("Building PCA model and transforming feature data\n")
            pca_model = train_dataset.pca(float(self.configuration['PCA_n_components']),
                                          "results/pcaModels/" + self.configuration['chosenDataset'] + "_pca_"
                                          + str(self.configuration['PCA_n_components']))
            train_PCA_components = train_dataset.pcaTransform(pca_model)
            test_PCA_components = test_dataset.pcaTransform(pca_model)
            train_matrix.computeMatrix(train_PCA_components)
            test_matrix.computeMatrix(train_PCA_components, test_PCA_components)
        print("\n")
        self.result_file.write("Train similarity matrix:\n" + str(train_matrix.matrix) + " \nShapes: " +
                               str(train_matrix.matrix.shape) + "\n\n")
        self.result_file.write("Test similarity matrix:\n" + str(test_matrix.matrix) + " \nShapes: " +
                               str(test_matrix.matrix.shape) + "\n\n")

        # train_matrix.buildSimilarityBoxPlot()
        # train_matrix.buildNeighborCountBoxPlotAndReport(float(self.configuration['minSimilarityValues']))

        print("\n COMPUTING TORCH GRAPHS \n")
        train_graph = train_matrix.generateTorchGraph(train_dataset, float(self.configuration['minSimilarityValues']),
                                                      torch_directml.device(),
                                                      path=self.configuration['pathPytorchGraphs'] +
                                                      self.configuration['chosenDataset'] + "_train_torch_graph.pkl")
        test_graph = test_matrix.generateTorchGraph(train_dataset, float(self.configuration['minSimilarityValues']),
                                                    torch_directml.device(), dataset_column=test_dataset,
                                                    path=self.configuration['pathPytorchGraphs'] +
                                                    self.configuration['chosenDataset'] + "_test_torch_graph.pkl")
        self.result_file.write("Train graph info:" + str(train_graph) + "\n\n")
        self.result_file.write("Test graph info:" + str(test_graph) + "\n\n")

