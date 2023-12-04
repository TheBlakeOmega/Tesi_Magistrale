import configparser
import numpy as np
from pipelineManager import PipeLineManager
import random

np.random.seed(12)
random.seed(12)


def datasetException(dataset_name):
    try:
        if dataset_name is None:
            raise TypeError()
        if not ((dataset_name == 'MALDROID') or (dataset_name == 'MALMEM')
                or (dataset_name == 'NSL_KDD') or (dataset_name == 'UNSWNUMERIC')):
            raise ValueError()
        return dataset_name
    except ValueError:
        print("Dataset not exist: must be MALDROID or MALMEM or NSL_KDD or UNSWNUMERIC")
    except TypeError:
        print("The name of dataset is null: use MALDROID or MALMEM or NSL_KDD or UNSWNUMERIC")


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('configuration.conf')
    configuration = config['SETTINGS']
    dataset = datasetException(configuration['chosenDataset'])
    dsConf = config[dataset]

    with open("results/result_" + dataset + "_" +
              configuration['convolutionalLayersNumber'] + "_conv_"
              + configuration['maxNeighbours'] + "_neighbors_"
              + configuration['chosenPipeline'] + ".txt", "w") as result_file:
        pManager = PipeLineManager(configuration, dsConf, configuration['chosenPipeline'], result_file)
        pManager.runPipeline()

        result_file.close()
