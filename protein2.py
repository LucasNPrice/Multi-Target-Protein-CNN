import pandas as pd
import numpy as np
import sys
from collections import Counter
import os
import glob
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from numpy import linalg as LA
from keras.models import Sequential
from keras.layers import Conv2D, Activation, Flatten, Dense
from sklearn.metrics import multilabel_confusion_matrix

# load data 
train = pd.read_csv('train.csv')
# train = pd.read_csv('train.csv').iloc[:50,:] # subset for building/debugging 
data_len = len(train)
img_dim = 512

# separate proteins in array by comma 
cat_train = []
for i in range(0, len(train.loc[:,'Target'])):
  print('Label: {} / {}'.format(i+1, len(train.loc[:,'Target'])))
  train.loc[i,'Target'] = (train.loc[i,'Target']).split(" ")
  cat_train = np.concatenate((cat_train, np.array(train.loc[i,'Target'])),axis=0)

# get unique protein set 
protein_set = set(cat_train)
protein_set = sorted([int(i) for i in protein_set])
protein_set = [str(i) for i in protein_set]

# map protein label to unique identifier 
protein_dict = {}
for i, j in enumerate(protein_set):
  protein_dict[j] = i

# create multi-hot array to represent the targets 
y_hot_data = np.zeros((len(train),len(protein_set)))
for i, j in enumerate(train.loc[:,'Target']):
  counts = Counter(train.iloc[i,1])
  for key in counts:
    y_hot_data[i,protein_dict[key]] = counts[key]

# check if any images contain more than 1 specific protein 
mult_proteins = []
for i,j in enumerate(y_hot_data):
  for k,l in enumerate(j):
    if l > 1:
      mult_proteins.append((i,j,k,l))

# load images 
img_dir = "/Users/lukeprice/tensorflow/Kaggle/train" 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
train_images = []; train_names = []
for count, file in enumerate(files): # 124288 image files, 31073 green image files 
  # print('file # {}'.format(count))
  if "green" in file:
    train_names.append(file[41:-10])
    print('Image: {} / {}'.format(len(train_names), data_len))
    img = cv2.imread(file)[0:img_dim,0:img_dim,:] ###################### remove slicing 
    train_images.append(img)
  if len(train_names) == data_len:
    break

# determine if all r,g,b intensities are the same for each pixel
equal_pix = 0; non_equal_pix = 0
for i, j in enumerate(train_images):
  print('Image Pt. 2: {} / {}'.format(i+1, data_len))
  for row in range(3):
    for col in range(3):
      val = j[row,col,0]
      rbg = j[row,col,:]
      if (all(pix == val for pix in rbg)):
        equal_pix += 1
      else:
        non_equal_pix += 1

# turn r,b,g images to greyscale if r,b,g intensities are all the same 
rbg_dim = train_images[0].shape[2]
if non_equal_pix == 0:
  rbg_dim = 1
  train_images = [img[:,:,rbg_dim] for img in train_images]

# scale train images and run pca
def pca_run(images, n_components, train_test):

  pca = PCA(n_components)
  var_exp_list = []
  var_iter_list = []
  for i, img in enumerate(images):
    print('PCA Image: {} / {}'.format(i+1, len(images)))
    img = StandardScaler().fit_transform(img)
    pca.fit(img) 
    if train_test == 0:
      v_exp = 0
      v_iter = 0
      while v_exp <= 0.99:
        v_exp += pca.explained_variance_ratio_[v_iter]
        v_iter += 1
      var_exp_list.append(v_exp)
      var_iter_list.append(v_iter)
    else:
      var_exp_list.append(sum(pca.explained_variance_ratio_))
      images[i] = pca.components_
  # images = principle components/eigenvectors; tranformed/reduced images 
  # var_exp_list = list of variance explained for each image 
  # var_iter_list = number of eigenvectors/values necessary for 99% var. explained 
  return((images, var_exp_list, var_iter_list))

# get best number of left eigenvectors (var. explained >= 99%)
train_pca = pca_run(train_images, img_dim, 0)
pca_dim_1 = int(np.ceil(np.mean(train_pca[2])))
test_pca = pca_run(train_images, pca_dim_1, 1)
train_images = test_pca[0]

# get best number of right eigenvectors (var. explained >= 99%)
train_images = [np.transpose(img, (1,0)) for img in train_images]
train_pca = pca_run(train_images, pca_dim_1, 0)
pca_dim_2 = int(np.ceil(np.mean(train_pca[2])))
test_pca = pca_run(train_images, pca_dim_2, 1)
train_images = test_pca[0]
train_images = [np.transpose(img, (1,0)) for img in train_images]
new_img_dim = (pca_dim_1,pca_dim_2)

# create sorted dataframe of image names and files
label_X_df = pd.DataFrame(
    {'File': train_names,
     'Image': train_images,
    })
del(train_images)
label_X_df = label_X_df.sort_values(by=['File'])

# # train and label_X_df are sorted in the same order; the image indexes are equal 
# # y_hot_data is also sorted in the same order as train and label_X_df
# create training and testing data 
image_names = label_X_df.loc[:,'File']
x_images = np.array(label_X_df.loc[:,'Image'])
del(label_X_df)

if rbg_dim == 1:
  stack = np.empty((data_len,*new_img_dim))
else:
  stack = np.empty((data_len,*new_img_dim,rbg_dim))
for i, img in enumerate(x_images):
  print('Stack: {} / {}'.format(i+1, len(x_images)))
  stack[i] = img
del(x_images)

train_index = np.random.choice(len(image_names), 
  int(np.ceil(len(image_names)/2)), 
  replace=False)
test_index = np.delete(np.arange(0,len(image_names),1), obj = train_index)

# # x_train.shape = (250,512,512,3) or (num_sample/2,num_rows,num_col,color_dim)
# # y_train.shape = (250,26); y_train[i].shape = (26,)
(x_train, y_train) = stack[train_index],y_hot_data[train_index]
(x_test, y_test) = stack[test_index], y_hot_data[test_index]
x_train = x_train.reshape(len(train_index),*new_img_dim,rbg_dim)
x_test = x_test.reshape(len(train_index),*new_img_dim,rbg_dim)
del(stack)

print(mult_proteins)
perc_reduced = (1-(np.multiply(*new_img_dim)/(img_dim**2)))*100
print('PCA Dimension: {} --- {} % compression'.format(
  new_img_dim, round(perc_reduced, 2)))
# print('Avg Variance Explained: {} %'.format(np.mean(test_pca[1]*100)))
# print('Variance Range: {} - {} %'.format(np.min(test_pca[1])*100,np.max(test_pca[1])*100))
print()
sys.exit()
#--------------------------------------------------------------------------------------
# create model
#--------------------------------------------------------------------------------------
model = Sequential()
model.add(Conv2D(input_shape = (*new_img_dim,rbg_dim),
  filters = 182, 
  kernel_size = 6, 
  activation = 'relu'))
model.add(Conv2D(filters = 96, 
  kernel_size = 3, 
  activation = 'relu'))
model.add(Flatten())
model.add(Dense(y_train.shape[1], activation = 'sigmoid'))
# if a protein i can be in an image multiple times, then ...
# ... use a non-truncated activation function (i.e. 'relu', 'linear', etc.)
# also, consider a custom activation which then rounds the outputs (i.e. 1.23 = 1)

# compile model using accuracy to measure model performance
# if a protein i can be in an image multiple times, then use categorical_cross_entropy 
model.compile(optimizer = 'adam', 
  loss = 'binary_crossentropy', 
  metrics = ['accuracy'])

# train model
model.summary()
model.fit(x_train, y_train, 
  validation_data = (x_test, y_test), 
  batch_size = 128,
  epochs = 3)

#--------------------------------------------------------------------------------------
# Get predictions and error rates 
#--------------------------------------------------------------------------------------
# predict test image proteins 
predictions = model.predict(x_test)
print(predictions.shape,"\n")
for i,pred in enumerate(predictions):
  for j in range(0,len(pred)):
    if pred[j] > 0.5:
      predictions[i,j] = 1
    else:
      predictions[i,j] = 0

# tn = 0,0; fn = 1,0; tp = 1,1; fp = 0,1
conf_mat = multilabel_confusion_matrix(y_test, predictions)
conf_mat_sum = np.zeros((2,2))
for mat in conf_mat:
  conf_mat_sum += mat

tn, fp, fn, tp = conf_mat_sum.flatten()
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)

print('-- Confusion Matrix --{} {}'.format('\n', conf_mat_sum))
print('True Positive Rate: {} {}False Positive Rate: {}'.format(tpr, '\n', fpr))








