import pandas as pd

"""
This class models a dataset loaded from storage, that belongs to the path built according to the dataset configuration
in configuration.conf file

Init Parameters
----------
:param dataset_configuration:
program dataset configuration
"""


class Dataset:

    def __init__(self, dataset_configuration):
        self.dataset_configuration = dataset_configuration
        self.features = None
        self.labels = None

    """
        This method load the dataset from storage
        Parameters
        ----------
        :param dataset_type:
        string that specifies if data belongs to train or test set
    """

    def loadDataset(self, dataset_type):
        self.features = pd.read_csv(self.dataset_configuration.get('path' + dataset_type + 'Dataset'))
        self.features.columns = self.features.columns.str.replace(' ', '')
        self.labels = pd.DataFrame(self.features[self.dataset_configuration.get('labelColumnName')])
        self.features.drop(columns=[self.dataset_configuration.get('labelColumnName')], inplace=True)
        print("LDS: " + dataset_type + ' data loaded:\n    columns:' + str(len(self.features.columns)) + '    rows:' +
              str(len(self.features.index)) + "\n\n")
