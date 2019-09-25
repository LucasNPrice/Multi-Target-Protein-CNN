import sys 
from tqdm import tqdm
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
from sklearn.decomposition import TruncatedSVD

# scale train images and run pca
# "
#   images = principle components/eigenvectors; tranformed/reduced images 
#   var_exp_list = list of variance explained for each image 
#   var_iter_list = number of eigenvectors/values necessary for 99% var. explained 
# "

# get number of pc's necessary to capture 'var_explain' variance of images
def train_pca(images, n_components, var_explain):

  pca = PCA(n_components)
  var_exp_list = []
  var_iter_list = []

  with tqdm(total = len(images)) as pbar:
    for i, img in enumerate(images):
      img = StandardScaler().fit_transform(img)
      pca.fit(img) 
      v_exp = 0
      v_iter = 0
      while v_exp <= var_explain:
        v_exp += pca.explained_variance_ratio_[v_iter]
        v_iter += 1
      var_exp_list.append(v_exp)
      var_iter_list.append(v_iter)
      pbar.update(1) 
  return (images, var_exp_list, var_iter_list)

# run pca and return images corresponding to 'var_explain' as captured by 'n_components'
def get_pca(images, n_components):

  pca = PCA(n_components)
  var_exp_list = []

  with tqdm(total = len(images)) as pbar:
    for i, img in enumerate(images):
      img = StandardScaler().fit_transform(img)
      pca.fit(img) 
      var_exp_list.append(sum(pca.explained_variance_ratio_))
      images[i] = pca.components_
      pbar.update(1) 
  return (images, var_exp_list)

def train_sparse_SVD(images, n_components, var_explain):

  svd = TruncatedSVD(n_components = n_components, n_iter=7, random_state=42)
  var_exp_list = []
  var_iter_list = []

  with tqdm(total = len(images)) as pbar:
    for i, img in enumerate(images):
      svd.fit(img)  
      v_exp = 0
      v_iter = 0
      while v_exp <= var_explain:
        v_exp += svd.explained_variance_ratio_[v_iter]
        v_iter += 1
      var_exp_list.append(v_exp)
      var_iter_list.append(v_iter)
      pbar.update(1) 
  return (images, var_exp_list, var_iter_list)

def get_sparse_SVD(images, n_components):

  svd = TruncatedSVD(n_components = n_components, n_iter=7, random_state=42)
  var_exp_list = []

  # with tqdm(total = len(images)) as pbar:
  for i, img in enumerate(images):
    svd.fit(img)  
    var_exp_list.append(np.sum(svd.explained_variance_ratio_))
    images[i] = svd.components_
    # pbar.update(1) 
  return (images, var_exp_list)