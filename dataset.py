import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pickle
import torch


class Dataset:
    """
    This class models a dataset loaded from storage, that belongs to the path passed as input

    Init Parameters
    ----------
    :param path:
    storage path from which load the dataset
    :param target:
    name of dataset's label column
    """

    def __init__(self, path, target):
        self.feature_data = pd.read_csv(path)
        self.feature_data.columns = self.feature_data.columns.str.replace(' ', '')
        self.label_data = pd.DataFrame(self.feature_data[target])
        self.feature_data.drop(columns=[target], inplace=True)

    def getLabelData(self):
        """
        This method returns dataset's label data
        Parameters
        ----------
        :return:
        dataset's label data
        """
        return self.label_data

    def getFeatureData(self):
        """
        This method returns dataset's feature data
        Parameters
        ----------
        :return:
        dataset's feature data
        """
        return self.feature_data

    def pca(self, n_components, path=None):
        """
        This method build a scikit-learn PCA model trained on dataset's feature data. If a path is passed, built
        model will be saved in that path
        Parameters
        ----------
        :param n_components:
        principal components to take from computed ones
        :param path:
        default None. Path in which PCA model will be saved
        :return:
        a scikit-learn PCA model trained on dataset's feature data
        """
        pca_model = PCA(n_components=n_components, svd_solver='full')
        pca_model = pca_model.fit(self.feature_data)
        if path is not None:
            with open(path, 'wb') as file:
                pickle.dump(pca_model, file)
        print("DPCA: PCA Object Params: ", pca_model.get_params())
        print("DPCA: PCA estimated principal components: ", pca_model.n_components_)
        return pca_model

    def pcaTransform(self, pca_model):
        """
        This method creates a new Dataframe transforming feature_data through input pca model
        Parameters
        ----------
        :param pca_model:
        a trained scikit-learn PCA model
        :return:
        Dataframe with principal components computed on feature_data
        """
        pca_data = self.feature_data.copy()
        pca_data = pca_model.transform(pca_data)
        pca_data = pd.DataFrame(pca_data, columns=pca_model.get_feature_names_out())
        return pca_data

    def computeMasks(self):
        """
        Compute masks of train or test and validation set
        :return:
        torch tensors representing masks of train or test and validation set
        """
        X, X_validation, y, y_validation = train_test_split(self.feature_data, self.label_data, test_size=0.2,
                                                            random_state=12, stratify=self.label_data)
        validation_mask = []
        mask = []
        for record in self.feature_data.index:
            if record in X.index:
                mask.append(True)
                validation_mask.append(False)
            else:
                mask.append(False)
                validation_mask.append(True)
        torch_validation_mask = torch.tensor(validation_mask, dtype=torch.bool)
        torch_mask = torch.tensor(mask, dtype=torch.bool)
        return torch_mask, torch_validation_mask
















