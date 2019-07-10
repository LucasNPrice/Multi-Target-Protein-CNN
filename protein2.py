import pandas as pd
import numpy as np
import sys
from collections import Counter
import os
import glob
import cv2

# load data 
train = pd.read_csv('train.csv')
# train = train.iloc[:500,:] # subset for building/debugging 

# separate proteins in array by comma 
cat_train = []
for i in range(0, len(train.loc[:,'Target'])):
  print(i)
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
  print(i)
  counts = Counter(train.iloc[i,1])
  for key in counts:
    y_hot_data[i,protein_dict[key]] = counts[key]

# load images 
img_dir = "/Users/lukeprice/tensorflow/Kaggle/train" 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
train_images = []; train_names = []
for count, file in enumerate(files): # 124288 image files, 31073 green image files 

  print('file # {}'.format(count))
  if "green" in file:
    train_names.append(file[41:-10])
    img = cv2.imread(file)
    train_images.append(img)
  # if len(train_names) == 500:
  #   break
  # # for building/debugging 

# create sorted dataframe of image names and files
label_X_df = pd.DataFrame(
    {'File': train_names,
     'Image': train_images,
    })
label_X_df = label_X_df.sort_values(by=['File'])

# train and label_X_df are sorted in the same order; the image indexes are equal 
# y_hot_data is also sorted in the same order as train and label_X_df

# create training and testing data 
image_names, x_images = label_X_df.loc[:,'File'], label_X_df.loc[:,'Image']

train_index = np.random.choice(len(image_names), 
  int(np.ceil(len(image_names)/2)), 
  replace=False)
test_index = np.delete(np.arange(1,len(image_names)+1,1), obj = train_index)

(x_train, y_train) = x_images[train_index], y_hot_data[train_index]
(x_test, y_test) = x_images[test_index], y_hot_data[test_index]

#--------------------------------------------------------------------------------------
# create model
