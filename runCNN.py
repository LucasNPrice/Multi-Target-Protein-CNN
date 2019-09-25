import CNN_batch as cnn

bcnn = cnn.CNN_batch(imgPath = '/home/ubuntu/protein_files/train_green',
  targetPath = '/home/ubuntu/protein_files/train.csv',
  batchsize = 24)

bcnn.train_model(epochs = 23)

bcnn.predict()

bcnn.eval_predictions()
