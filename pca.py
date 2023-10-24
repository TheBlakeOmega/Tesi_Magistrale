from sklearn.decomposition import PCA
import pandas as pd
import joblib

"""
Build and fit PCA model
Parameters
----------
:param data:
dataSet that will be transformed
Returns
----------
:pcaModel:
PCA model computed
:pcaList:
list of names of principals components computed
"""


def buildPCAModel(data, dataset_name, components):
    pca_model = PCA(n_components=components, svd_solver='full')
    pca_model = pca_model.fit(data)
    joblib.dump(pca_model, "results/pcaModels/" + dataset_name + "_pca_" + str(components))
    print("BPCA: PCA Object Params: ", pca_model.get_params())
    print("BPCA: PCA estimated principal components: ", pca_model.n_components_)
    return pca_model


"""
Transform data in their principal components through an input PCA model
Parameters
----------
:param data:
dataSet that will be transformed
:param pcaModel:
a PCA model 
:param pcaList:
list of principals components in which transform data
Returns
----------
:pcaData:
data transformed
"""


def applyPCA(data, pca_model):
    pca_data = data.copy()
    pca_data = pca_model.transform(pca_data)
    pca_data = pd.DataFrame(pca_data, columns=pca_model.get_feature_names_out())
    return pca_data
