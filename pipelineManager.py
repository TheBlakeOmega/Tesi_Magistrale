from dataset import Dataset
from cosineSimilarityMatrix import CosineSimilarityMatrixTrain, CosineSimilarityMatrixTest
import traceback
import pickle
import torch
from graphConvolutionalNetworkClassifier import GraphNetwork
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from hyperopt import tpe, hp, Trials, fmin


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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        elif self.chosen_pipeline == 'TRAIN_GCN':
            try:
                self._trainGraphConvolutionalNetwork()
                print("GCN trained successfully.")
                self.result_file.write("GCN trained successfully.")
            except Exception as e:
                traceback.print_exc()
                raise Exception("Error during GCN train")
        elif self.chosen_pipeline == 'COMPUTE_EVALUATION_METRICS_TRAIN_VALIDATION':
            try:
                self._computeEvaluationMetricsOnTrainAndValidationSet()
                print("Evaluation metrics computed successfully.")
                self.result_file.write("Evaluation metrics computed successfully.")
            except Exception as e:
                traceback.print_exc()
                raise Exception("Error during evaluation metrics computation")
        elif self.chosen_pipeline == 'TEST_GCN':
            try:
                self._testGraphConvolutionalNetwork()
                print("GCN tested successfully.")
                self.result_file.write("GCN tested successfully.")
            except Exception as e:
                traceback.print_exc()
                raise Exception("Error during GCN test")
        elif self.chosen_pipeline == 'OPTIMIZE_GCN':
            try:
                self._optimizeGraphConvolutionalNetwork()
                print("GCN's parameters optimized successfully")
                self.result_file.write("GCN's parameters optimized successfully.")
            except Exception as e:
                traceback.print_exc()
                raise Exception("Error during GCN optimization")

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
        train_matrix = CosineSimilarityMatrixTrain()
        test_matrix = CosineSimilarityMatrixTest()
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
        train_graph_save_path = (self.configuration['pathPytorchGraphs'] + self.configuration['chosenDataset']
                                 + "_" + self.configuration['minSimilarityValues'] + "_similarity_"
                                 + "_" + self.configuration['maxNeighbours'] + "_neighbors_train_torch_graph.pkl")
        train_graph = train_matrix.generateTrainTorchGraph(train_dataset,
                                                           float(self.configuration['minSimilarityValues']),
                                                           int(self.configuration['maxNeighbours']),
                                                           self.device,
                                                           path=train_graph_save_path)
        self.result_file.write("Train graph info:" + str(train_graph) + "\n\n")

    def _trainGraphConvolutionalNetwork(self):
        """
        This method runs the pipeline to train and serialize GCN model
        """
        load_path = (self.configuration['pathPytorchGraphs'] + self.configuration['chosenDataset']
                     + "_" + self.configuration['minSimilarityValues'] + "_similarity_"
                     + "_" + self.configuration['maxNeighbours'] + "_neighbors_train_torch_graph.pkl")
        with open(load_path, 'rb') as file:
            print("TGCN: Loading train graph from " + load_path)
            train_graph = pickle.load(file)
            print("TGCN: Train graph loaded")
            file.close()
        model = GraphNetwork()
        space = {
            'epochs': 200,
            'earlyStoppingThresh': 200
        }
        print("Starting GCN training on " + torch.cuda.get_device_name(0) + " " + str(self.device))
        save_path = (self.configuration['pathModels'] + self.configuration['chosenDataset'] +
                     "_" + self.configuration['minSimilarityValues'] + "_similarity_" +
                     self.configuration['convolutionalLayersNumber'] + "_conv_"
                     + self.configuration['maxNeighbours'] + "_neighbors_trained_GCN.pkl")
        start_train_time = np.datetime64(datetime.now())
        scores = model.train(train_graph, space, self.device, save_path)
        end_train_time = np.datetime64(datetime.now())
        self.result_file.write("Trained model result:" + "\nLoss: " + str(scores['train_loss']) +
                               "\nAccuracy: " + str(scores['train_accuracy']) +
                               "\nTrain time: " + str(end_train_time - start_train_time) + "\n")

    def _testGraphConvolutionalNetwork(self):
        """
        This method runs the pipeline to compute evaluation metrics on test set
        """
        print("TeGCN: Loading test dataset from " + self.ds_configuration['pathTestDataset'])
        test_dataset = Dataset(self.ds_configuration['pathTestDataset'],
                               self.ds_configuration['labelColumnName'])
        print("TeGCN: Dataset Loaded")
        print("TeGCN: Loading test similarity matrix from " + self.configuration['pathSimilarityMatrices'] +
              self.configuration['chosenDataset'] + "_test_similarity_matrix.pkl")
        test_matrix = CosineSimilarityMatrixTest()
        test_matrix.loadMatrix(self.configuration['pathSimilarityMatrices'] + self.configuration['chosenDataset'] +
                               "_test_similarity_matrix.pkl")
        print("TeGCN: Similarity Matrix Loaded")
        graph_load_path = (self.configuration['pathPytorchGraphs'] + self.configuration['chosenDataset']
                           + "_" + self.configuration['minSimilarityValues'] + "_similarity_"
                           + "_" + self.configuration['maxNeighbours'] + "_neighbors_train_torch_graph.pkl")
        with open(graph_load_path, 'rb') as file:
            print("TeGCN: Loading train graph from " + graph_load_path)
            train_graph = pickle.load(file)
            print(train_graph)
            print("TeGCN: Train graph loaded")
            file.close()
        print("TeGCN: Loading model from " + self.configuration['pathModels'] + self.configuration['chosenDataset'] +
              "_trained_GCN.pkl")
        model = GraphNetwork()
        model.loadModel(self.configuration['pathModels'] + self.configuration['chosenDataset'] +
                        "_" + self.configuration['minSimilarityValues'] + "_similarity_" +
                        self.configuration['convolutionalLayersNumber'] + "_conv_"
                        + self.configuration['maxNeighbours'] + "_neighbors_trained_GCN.pkl",
                        self.device)
        print("TeGCN: Model Loaded")

        predictions = []
        counter = 1
        start_test_time = np.datetime64(datetime.now())
        for test_example_index, test_example in test_dataset.getFeatureData().iterrows():
            test_graph = test_matrix.addTestExampleToGraph(float(self.configuration['minSimilarityValues']),
                                                           train_graph, test_example, test_example_index,
                                                           int(self.configuration['maxNeighbours']), self.device)
            test_mask = [False for i in range(len(test_graph.x.tolist()))]
            test_mask[-1] = True
            test_mask = torch.tensor(test_mask, dtype=torch.bool)
            example_prediction = model.test(test_graph, self.device, test_mask).tolist()
            predictions.extend(example_prediction)

            if counter % 10 == 0:
                print("TeGCN: " + str(counter) + " examples computed")
            counter += 1
        end_test_time = np.datetime64(datetime.now())

        labels = list(set(test_dataset.getLabelData()[self.ds_configuration['labelColumnName']]))
        print("TeGCN: Computing confusion matrix")
        test_confusion_matrix = confusion_matrix(
            test_dataset.getLabelData()[self.ds_configuration['labelColumnName']],
            predictions, labels=labels)
        print("TeGCN: Computing classification report")
        test_classification_report = classification_report(
            test_dataset.getLabelData()[self.ds_configuration['labelColumnName']], predictions, digits=3)
        self.result_file.write("Test confusion matrix:\n")
        self.result_file.write(str(test_confusion_matrix) + "\n")
        self.result_file.write("Test classification report:\n")
        self.result_file.write(str(test_classification_report) + "\n")
        self.result_file.write("Test computation time:\n")
        self.result_file.write(str(end_test_time - start_test_time) + "\n")
        test_confusion_matrix_plot = ConfusionMatrixDisplay(test_confusion_matrix, display_labels=labels)
        test_confusion_matrix_plot.plot()
        plt.title("Test Confusion Matrix " + self.configuration['chosenDataset'])
        plt.savefig(self.configuration['chosenDataset'] + "_" +
                    self.configuration['minSimilarityValues'] + "_similarity_" +
                    self.configuration['convolutionalLayersNumber'] + "_conv_" +
                    self.configuration['maxNeighbours'] + "_neighbors_test_confusion_matrix.png")
        plt.close()

    def _computeEvaluationMetricsOnTrainAndValidationSet(self):
        """
        This method runs the pipeline to compute evaluation metrics on train and validation set
        """
        graph_load_path = (self.configuration['pathPytorchGraphs'] + self.configuration['chosenDataset']
                           + "_" + self.configuration['minSimilarityValues'] + "_similarity_"
                           + "_" + self.configuration['maxNeighbours'] + "_neighbors_train_torch_graph.pkl")
        with open(graph_load_path, 'rb') as file:
            print("CCM: Loading train graph from " + graph_load_path)
            input_graph = pickle.load(file)
            print(input_graph)
            print("CCM: Train graph loaded")
            file.close()
        model = GraphNetwork()
        model.loadModel(self.configuration['pathModels'] + self.configuration['chosenDataset'] +
                        "_" + self.configuration['minSimilarityValues'] + "_similarity_" +
                        self.configuration['convolutionalLayersNumber'] + "_conv_"
                        + self.configuration['maxNeighbours'] + "_neighbors_trained_GCN.pkl",
                        self.device)
        print("CCM: Computing predictions")
        predictions = model.test(input_graph, self.device)
        print("CCM: Computing confusion matrices")
        labels = list(set(input_graph.y.squeeze().tolist()))
        train_confusion_matrix = confusion_matrix(input_graph.y[input_graph.train_mask.to(self.device)].squeeze()
                                                  .tolist(), predictions[input_graph.train_mask.to(self.device)]
                                                  .tolist(), labels=labels)
        validation_confusion_matrix = confusion_matrix(input_graph.y[input_graph.validation_mask.to(self.device)]
                                                       .squeeze().tolist(), predictions[input_graph.validation_mask
                                                       .to(self.device)].tolist(), labels=labels)
        train_classification_report = classification_report(
            input_graph.y[input_graph.train_mask.to(self.device)].squeeze()
            .tolist(), predictions[input_graph.train_mask.to(self.device)]
            .tolist(), digits=3)
        validation_classification_report = classification_report(
            input_graph.y[input_graph.validation_mask.to(self.device)].squeeze()
            .tolist(), predictions[input_graph.validation_mask.to(self.device)]
            .tolist(), digits=3)

        self.result_file.write("Train confusion matrix:\n")
        self.result_file.write(str(train_confusion_matrix) + "\n")
        self.result_file.write("Train classification report:\n")
        self.result_file.write(str(train_classification_report) + "\n")
        self.result_file.write("Validation confusion matrix:\n")
        self.result_file.write(str(validation_confusion_matrix) + "\n")
        self.result_file.write("Validation classification report:\n")
        self.result_file.write(str(validation_classification_report) + "\n")
        train_confusion_matrix_plot = ConfusionMatrixDisplay(train_confusion_matrix, display_labels=labels)
        train_confusion_matrix_plot.plot()
        plt.title("Train Confusion Matrix " + self.configuration['chosenDataset'])
        plt.savefig(self.configuration['chosenDataset'] + "_" +
                    self.configuration['minSimilarityValues'] + "_similarity_" +
                    self.configuration['convolutionalLayersNumber'] + "_conv_" +
                    self.configuration['maxNeighbours'] + "_neighbors_train_confusion_matrix.png")
        plt.close()
        validation_confusion_matrix_plot = ConfusionMatrixDisplay(validation_confusion_matrix, display_labels=labels)
        validation_confusion_matrix_plot.plot()
        plt.title("Validation Confusion Matrix " + self.configuration['chosenDataset'])
        plt.savefig(self.configuration['chosenDataset'] + "_" +
                    self.configuration['minSimilarityValues'] + "_similarity_" +
                    self.configuration['convolutionalLayersNumber'] + "_conv_" +
                    self.configuration['maxNeighbours'] + "_neighbors_validation_confusion_matrix.png")
        plt.close()

    def _optimizeGraphConvolutionalNetwork(self):
        """
        This method runs the pipeline to optimize train's parameters
        """
        load_path = (self.configuration['pathPytorchGraphs'] + self.configuration['chosenDataset']
                     + "_" + self.configuration['minSimilarityValues'] + "_similarity_"
                     + "_" + self.configuration['maxNeighbours'] + "_neighbors_train_torch_graph.pkl")
        with open(load_path, 'rb') as file:
            print("OGCN: Loading train graph from " + load_path)
            train_graph = pickle.load(file)
            print("OGCN: Train graph loaded")
            file.close()
        model = GraphNetwork()
        save_result_path = (self.configuration['pathHyperopt'] + self.configuration['chosenDataset']
                            + "_" + self.configuration['minSimilarityValues'] + "_similarity_"
                            + self.configuration['convolutionalLayersNumber'] + "_conv_"
                            + self.configuration['maxNeighbours'] + "_neighbors_trained_GCN_result.csv")
        save_model_path = (self.configuration['pathModels'] + self.configuration['chosenDataset'] + "_" +
                           self.configuration['minSimilarityValues'] + "_similarity_" +
                           self.configuration['convolutionalLayersNumber'] + "_conv_" +
                           self.configuration['maxNeighbours'] + "_neighbors_trained_GCN.pkl")
        space = {
            'input_graph': train_graph,
            'device': self.device,
            'save_results_path': save_result_path,
            'save_model_path': save_model_path,
            'conv_layers': int(self.configuration['convolutionalLayersNumber']),
            'learning_rate': hp.uniform("learning_rate", 0.0001, 0.001),
            'batch_size': hp.choice("batch", [32, 64, 128, 256, 512]),
            'epochs': int(self.configuration['trainEpochs']),
            'earlyStoppingThresh': int(self.configuration['earlyStoppingThresh'])
        }
        for i in range(int(self.configuration['convolutionalLayersNumber'])):
            space['dropout_' + str(i + 1)] = hp.uniform('dropout_' + str(i + 1), 0, 1)

        print("OGCN: Starting GCN optimization on " + torch.cuda.get_device_name(0) + " " + str(self.device))
        trials = Trials()
        fmin(model.optimizeParameters, space, trials=trials, algo=tpe.suggest, max_evals=20)
