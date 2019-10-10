import numpy as np
import os
from collections import Counter
import cv2
import matplotlib.pyplot as plt

import imageProcessor as imp
import lossFuns

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv2D, Activation, Flatten, Dense, MaxPooling2D, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import multilabel_confusion_matrix
from keras.constraints import maxnorm


class Train_on_Batch():

  def __init__(self, imgPath, targetPath, batchsize, train_val_test = (0.6,0.2,0.2)):

    self.imgPath = imgPath
    self.targetPath = targetPath
    self.img_files = sorted(os.listdir(self.imgPath))
    self.img_ids = [filename[0:-10] for filename in self.img_files]

    self.batchsize = batchsize
    self.train_test_split(train_val_test)
    self.img_dim = cv2.imread(self.imgPath + '/' + self.img_files[0]).shape
    self.compile_model()

    self.aug = ImageDataGenerator(rotation_range = 20, zoom_range = 0.15, 
      width_shift_range = 0.2, height_shift_range = 0.2, 
      shear_range = 0.15, horizontal_flip = True, 
      fill_mode = 'nearest')


  def train_test_split(self, train_val_test):
    # get unique proteins in data 
    IDs = []
    targets = []
    self.protein_set = set()
    num_images = 0
    with open(self.targetPath) as f:
      next(f)
      for line in f:
        ID, target = line.strip().split(",")
        target = target.split(" ")
        IDs.append(ID)
        targets.append(target)
        [self.protein_set.add(l) for l in target]
        num_images += 1
    f.close()

    # update protein set
    self.protein_set = sorted([int(i) for i in self.protein_set])
    self.protein_set = [str(i) for i in self.protein_set]
    # map protein label to unique identifier 
    self.protein_dict = {}
    for i, j in enumerate(self.protein_set):
      self.protein_dict[j] = i

    # shuffle data 
    ID_Target = np.vstack((IDs, targets)).transpose()
    np.random.shuffle(ID_Target)

    # turn targets to 1-hots
    y_hots = np.zeros((len(targets), len(self.protein_set)))
    for i, labels in enumerate(ID_Target[:,1]):
      counts = Counter(labels)
      for key in counts:
        y_hots[i,self.protein_dict[key]] = counts[key]
      ID_Target[i,1] = y_hots[i,:]

    # split data into train, validation, and test sets 
    self.train, self.validation, self.test = np.split(ID_Target, 
      [int(train_val_test[0]*num_images), 
      int(num_images*(np.add(*train_val_test[0:2])))])

    # get ID dictionary 
    self.IDs = {
      'train': self.train[:,0],
      'validation': self.validation[:,0],
      'test': self.test[:,0]
    }

    # get target dictionary 
    self.targets = {}
    for i in range(0,len(ID_Target)):
      self.targets[ID_Target[i,0]] = ID_Target[i,1]

    print('\nTotal Number of Images: {}'.format(num_images))
    print('Training Images: {}'.format(len(self.IDs['train'])))
    print('Validation Images: {}'.format(len(self.IDs['validation'])))
    print('Testing Images: {}'.format(len(self.IDs['test'])))


  def data_generator(self, data, mode = 'train', aug = None):

    file_extension = '_green.png'
    while True:
      images = []
      targets = []
      for i in range(0,len(data)):
        if len(images) == self.batchsize:
          targets = np.array(targets)
          img = imp.Image_Processor(images)
          img.greyscale()
          img.normalize()
          img.resize(x_perc = 0.4, y_perc = 0.4)
          self.img_dim = img.img_dim
          images = img.images
          images = np.array(images).reshape(len(targets),*img.img_dim,1)

          if aug is not None:
            (images, targets) = next(aug.flow(images,
              targets, batch_size = self.batchsize))

          yield (images, targets)
          if mode == 'predict':
            if self.test_targets == []:
              self.test_targets = targets
            else:
              self.test_targets = np.concatenate((self.test_targets, targets), axis = 0)
          images = []
          targets = []
          tid = []
          continue

        target_id = data[i]
        target = self.targets[target_id]
        image = cv2.imread(self.imgPath + '/' + target_id + file_extension)
        images.append(image)
        targets.append(target)


  def compile_model(self):

    optimizer = optimizers.Adam(lr=0.0005)
    self.model = Sequential()

    self.model.add(Conv2D(input_shape = (204, 204, 1),
      filters = 64, 
      kernel_size = 7,
      activation = 'relu'))
    self.model.add(BatchNormalization())
    self.model.add(MaxPooling2D(pool_size=(2, 2)))
    self.model.add(Dropout(0.2))

    self.model.add(Conv2D(filters = 128, 
      kernel_size = 3,
      activation = 'relu'))
    self.model.add(BatchNormalization())

    self.model.add(Conv2D(filters = 256, 
      kernel_size = 3,
      activation = 'relu'))
    self.model.add(BatchNormalization())
    self.model.add(MaxPooling2D(pool_size=(2, 2)))
    self.model.add(Dropout(0.2))

    self.model.add(Flatten())
    self.model.add(Dense(int(204), activation = 'relu', kernel_constraint = maxnorm(3)))
    self.model.add(Dense(len(self.protein_set), activation = 'sigmoid'))

    self.model.compile(optimizer = optimizer, 
      loss=[lossFuns.customFocalLoss], 
      metrics = ['accuracy', lossFuns.f1, lossFuns.f1_new])


  def train_model(self, epochs):
      # initialize both the training and testing image generators
      train_generator = self.data_generator(data = self.IDs['train'], 
        aug = self.aug)
      validation_generator = self.data_generator(data = self.IDs['validation'])

      self.model.summary()
      self.model.fit_generator(
        generator = train_generator,
        steps_per_epoch = len(self.IDs['train']) // self.batchsize,
        validation_data = validation_generator,
        validation_steps = len(self.IDs['validation']) // self.batchsize,
        epochs = epochs)


  def predict(self):

    self.test_targets = []
    test_generator = self.data_generator(data = self.IDs['test'], 
      mode = 'predict')
    self.predictions = self.model.predict_generator(test_generator, 
      steps = len(self.IDs['test']) // self.batchsize)


  def eval_predictions(self):

    for i, pred in enumerate(self.predictions):
      for j in range(0,len(pred)):
        if pred[j] >= 0.5:
          self.predictions[i,j] = 1
        else:
          self.predictions[i,j] = 0

    # tn = 0,0; fn = 1,0; tp = 1,1; fp = 0,1
    conf_mat = multilabel_confusion_matrix(self.test_targets, self.predictions)
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
