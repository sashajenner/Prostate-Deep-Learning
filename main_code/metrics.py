#custom metrics i.e. specificity and sensitivity

import keras.backend as K
import numpy as np
import tensorflow as tf

#ouput binary confusion matrix

def confusion_matrix(y_pred, y_true):
    not_true = 1 - y_true
    not_pred = 1 - y_pred
    
    FP = K.sum(not_true * y_pred).eval(session = tf.Session())
    TN = K.sum(not_true * not_pred).eval(session = tf.Session())

    FN = K.sum(y_true * not_pred).eval(session = tf.Session())
    TP = K.sum(y_true * y_pred).eval(session = tf.Session())
    
    print('''
       BINARY CONFUSION MATRIX
	      +-----------------+
    	      |    predicted    |
    +---------+-----------------+
    |actually |True	|False	|
    |---------|---------|-------|
    |True     |{}	|{}	|
    |False    |{}	|{}	|
    +---------+---------+-------+
    '''.format(TP, FN, FP, TN))

#custom specificity/sensitivity due to keras not having these

def spec(y_pred, y_true):

  """
  specificity:
  y_pred = matrix of labels that are predicted to be true
  y_true = labels that are true by the GT
  not_true = labels that are not true by the GT
  not_pred = labels that are predicted to be not true
  return spec
  """

  not_true = 1 - y_true

  not_pred = 1 - y_pred

  FP = K.sum(not_true * y_pred)

  TN = K.sum(not_true * not_pred)

  spec = TN / (TN + FP)

  return spec


def sens(y_pred, y_true):

  """
  sensitivity:
  y_pred = matrix of labels that are predicted to be true
  y_true = labels that are true by the GT
  not_true = labels that are not true by the GT
  not_pred = labels that are predicted to be not true
  return sens
  """

  # subtracting the tensors y_true and y_pred from 1 should give their complement tensors

  # since this is a binary classification

  not_true = 1 - y_true

  not_pred = 1 - y_pred

  FN = K.sum(y_true * not_pred)
  TP = K.sum(y_true * y_pred)

  sens = TP / (TP + FN)

  return sens
