import tensorflow as tf
import keras.backend as K
import sys

#---------------------------------------------------------------------------
# Keras focal loss
def KerasFocalLoss(target, y_hat): 
  
  gamma = 2.0
  y_hat = tf.cast(y_hat, tf.float32)
  max_val = K.relu(-y_hat)
  loss = y_hat - y_hat * target + max_val + K.log(K.exp(-max_val) + K.exp(-y_hat - max_val))
  invprobs = tf.log_sigmoid(-y_hat * (target * 2.0 - 1.0))
  loss = K.exp(invprobs * gamma) * loss
  
  return K.mean(K.sum(loss, axis=1))

#---------------------------------------------------------------------------
# metric 
def f1(y_true, y_pred):

  y_pred = K.round(y_pred)
  tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
  tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
  fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
  fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

  p = tp / (tp + fp + K.epsilon())
  r = tp / (tp + fn + K.epsilon())

  f1 = 2*p*r / (p+r+K.epsilon())
  f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
  return K.mean(f1)

# loss function
def f1_loss(y_true, y_pred):
  tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
  tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
  fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
  fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

  p = tp / (tp + fp + K.epsilon())
  r = tp / (tp + fn + K.epsilon())

  f1 = 2*p*r / (p+r+K.epsilon())
  f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
  return 1 - K.mean(f1)
  
#---------------------------------------------------------------------------
def f1_new(y_true, y_pred):
  def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

  def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
  precision = precision(y_true, y_pred)
  recall = recall(y_true, y_pred)
  return 2*((precision*recall)/(precision+recall+K.epsilon()))
