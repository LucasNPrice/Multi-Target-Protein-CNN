# batch
import os 
from copy import deepcopy
import cv2
import numpy as np
from collections import Counter

import imageProcessor as imp
import lossFuns

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv2D, Activation, Flatten, Dense, MaxPooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import multilabel_confusion_matrix
from keras.constraints import maxnorm

class CNN_batch():

  def __init__(self, imgPath, targetPath, batchsize):

    self.imgPath = imgPath
    self.targetPath = targetPath
    self.img_files = sorted(os.listdir(self.imgPath))
    self.img_ids = [filename[0:-10] for filename in self.img_files]

    self.batchsize = batchsize
    self.num_train_images = 0
    # self.num_test_images = 0

    self.protein_set = set()
    self.protein_dict = {}
    self.test_labels = []
    self.get_target_sets()
    self.compile_model()

    self.aug = ImageDataGenerator(rotation_range = 20, zoom_range = 0.15, 
      width_shift_range = 0.2, height_shift_range = 0.2, 
      shear_range = 0.15, horizontal_flip = True, 
      fill_mode = 'nearest')

  def _update_targets(self, targets):

    if self.test_labels == []:
      self.test_labels = deepcopy(targets)
    else:
      self.test_labels = np.concatenate((self.test_labels, deepcopy(targets)), axis = 0)

  def get_target_sets(self):
    
    # get unique proteins in data 
    with open(self.targetPath) as f:
      next(f)
      for line in f:
        label = line.strip().split(",")[1].split(" ")
        [self.protein_set.add(l) for l in label]
        self.num_train_images += 1
    f.close()

    self.protein_set = sorted([int(i) for i in self.protein_set])
    self.protein_set = [str(i) for i in self.protein_set]
    # map protein label to unique identifier 
    for i, j in enumerate(self.protein_set):
      self.protein_dict[j] = i

  def data_generator(self, mode = "train", aug = None):

    file_extension = '_green.png'
    while True:
      with open(self.targetPath, 'r', newline='') as f:
        next(f)
        images = []
        targets = []
        tid = []
        for line in f:
          if len(images) == self.batchsize:
            y_hot_batch = np.zeros((len(targets), len(self.protein_set)))
            for i, labels in enumerate(targets):
              counts = Counter(labels)
              for key in counts:
                y_hot_batch[i,self.protein_dict[key]] = counts[key]

            img = imp.Image_Processor(images)
            img.greyscale()
            img.normalize()
            img.getSVD(n_components = 192)
            img.transpose()
            img.getSVD(n_components = 191)
            img.transpose()
            images = img.images
            images = np.array(images).reshape(self.batchsize,*img.img_dim,1)

            if aug is not None:
              (images, y_hot_batch) = next(aug.flow(images,
                np.array(y_hot_batch), batch_size = self.batchsize))

            yield (images, y_hot_batch)
            if mode == 'eval':
              self._update_targets(y_hot_batch)
            images = []
            targets = []
            tid = []
            continue

          line = line.strip().split(",")
          target_id = line[0]
          target = line[1].split(" ")
          if target_id in set(self.img_ids):
            tid.append(target_id)
            image = cv2.imread(self.imgPath + '/' + target_id + file_extension)
            images.append(image)
            targets.append(target)

  def compile_model(self):

    optimizer = optimizers.Adam(lr=0.0005)

    self.model = Sequential()
    self.model.add(Conv2D(input_shape = (192, 191, 1),
      filters = 64, 
      kernel_size = 3,
      activation = 'relu'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))
    self.model.add(Dropout(0.2))

    self.model.add(Conv2D(filters = 128, 
      kernel_size = 3,
      activation = 'relu'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))
    self.model.add(Dropout(0.2))

    self.model.add(Conv2D(filters = 256, 
      kernel_size = 3,
      activation = 'relu'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))
    self.model.add(Dropout(0.2))

    self.model.add(Flatten())
    self.model.add(Dense(int(512), activation = 'relu', kernel_constraint = maxnorm(3)))
    self.model.add(Dense(len(self.protein_set), activation = 'sigmoid'))

    self.model.compile(optimizer = optimizer, 
      loss = lossFuns.f1_loss, 
      metrics = ['accuracy', lossFuns.f1])

  def train_model(self, epochs):

    # initialize both the training and testing image generators
    trainGen = self.data_generator(mode = 'train', aug = self.aug)
    testGen = self.data_generator(mode = 'train', aug = None)

    self.model.summary()
    self.model.fit_generator(
      trainGen,
      steps_per_epoch = len(self.img_files) // self.batchsize,
      validation_data = testGen,
      validation_steps = len(self.img_files) // self.batchsize,
      epochs = epochs,
      shuffle = True)

  def predict(self):

    testGen = self.data_generator(mode = 'eval', aug = None)
    # self.predictions = self.model.predict_generator(testGen, steps = self.num_train_images // self.batchsize)
    self.predictions = self.model.predict_generator(testGen, steps = len(self.img_files) // self.batchsize)


  def eval_predictions(self):
   
    # pred_length = len(self.predictions)
    # self.test_labels = self.test_labels[0:pred_length]

    for i, pred in enumerate(self.predictions):
      for j in range(0,len(pred)):
        if pred[j] >= 0.5:
          self.predictions[i,j] = 1
        else:
          self.predictions[i,j] = 0

    # tn = 0,0; fn = 1,0; tp = 1,1; fp = 0,1
    conf_mat = multilabel_confusion_matrix(self.test_labels, self.predictions)
    conf_mat_sum = np.zeros((2,2))
    for mat in conf_mat:
      conf_mat_sum += mat

    tn, fp, fn, tp = conf_mat_sum.flatten()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))

    print(' ----- Confusion Matrix -----{}  {} {}'.format('\n', conf_mat_sum, '\n'))
    print('True Positive Rate: {} {}False Positive Rate: {}'.format(round(tpr,3), '\n', round(fpr,3)))
    print('F1 Score: {}'.format(round(f1_score,3)))
