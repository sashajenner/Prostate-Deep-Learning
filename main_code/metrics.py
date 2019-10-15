#custom metrics i.e. specificity and sensitivity

import keras.backend as K

#custom specificty/sensitivity due to keras not having these

def specificity(x_pred, x_true):

  """
  specificity:
  x_pred = matrix of labels that are predicted to be true
  x_true = labels that are true by the GT
  not_true = labels that are not true by the GT
  not_pred = labels that are predicted to be not true
  return spec
  """

  not_true = 1-x_true

  not_pred = 1-x_pred

  FP = K.sum(not_true * x_pred)

  TN = K.sum(not_true * not_pred)

  spec = TN/(TN+FP)

  return spec


def sensitivity(x_pred, x_true):

  """
  sensitivity:
  x_pred = matrix of labels that are predicted to be true
  x_true = labels that are true by the GT
  not_true = labels that are not true by the GT
  not_pred = labels that are predicted to be not true
  return sens
  """

  # subtracting the tensors x_true and x_pred from 1 should give their complement tensors

  # since this is a binary classification

  not_true = 1-x_true

  not_pred = 1-x_pred

  FN = K.sum(not_true * x_pred)

  TP = K.sum(x_true * x_pred)

  sens = TP/(TP + FN)

  return sens
