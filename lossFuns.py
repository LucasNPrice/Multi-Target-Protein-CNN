import tensorflow as tf
import keras.backend as K

def customFocalLoss(target, y_hat):
  """ if y = 1, p = p, alpha = alpha
  else if y = 0, p = 1 - p, alpha = 1 - alpha
  form: -alpha * (1-p)^gamma * log(p) """
  gamma = 2.0
  alpha = 0.25

  zeros = tf.equal(target, 0)
  # alpha = tf.constant(alpha, shape = tf.shape(target))  
  alpha = tf.fill(tf.shape(target), alpha)
  alpha = tf.where(zeros, 1-alpha, alpha)

  y_hat = tf.cast(y_hat, tf.float32)
  target = tf.cast(target, tf.float32)

  # jitter data to range (0,1)
  zeros = tf.equal(y_hat, 0)
  y_hat = tf.where(zeros, y_hat+K.epsilon(), y_hat)
  ones = tf.equal(y_hat, 1)
  y_hat = tf.where(ones, y_hat-K.epsilon(), y_hat)

  relative_target_probs = tf.where(zeros, 1-y_hat, y_hat)
  loss = (-alpha * (1-relative_target_probs)**gamma) * tf.log(relative_target_probs)
  return K.mean(K.sum(loss, axis = 1))

#---------------------------------------------------------------------------
# Keras focal loss
# retrieved from https://www.kaggle.com/rejpalcz/focalloss-for-keras
def KerasFocalLoss(target, y_hat): 
  
  gamma = 2.0
  y_hat = tf.cast(y_hat, tf.float32)
  # max_val = K.clip(-input, 0, 1)
  max_val = K.relu(-y_hat)
  loss = y_hat - y_hat * target + max_val + K.log(K.exp(-max_val) + K.exp(-y_hat - max_val))
  invprobs = tf.log_sigmoid(-y_hat * (target * 2.0 - 1.0))
  loss = K.exp(invprobs * gamma) * loss
  return K.mean(K.sum(loss, axis=1))

#---------------------------------------------------------------------------
# Metrics
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

# second metric for F1
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
