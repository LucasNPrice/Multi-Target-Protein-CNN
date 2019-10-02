import train_on_batch

myCNN = train_on_batch.Train_on_Batch(
  imgPath = '/home/ubuntu/protein_files/train_green_copy',
  targetPath = '/home/ubuntu/protein_files/train.csv',
  batchsize = 24)

myCNN.train_model(epochs = 23)

myCNN.predict()

myCNN.eval_predictions()
