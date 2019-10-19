# Custom metrics i.e. specificity, sensitivity and confusion matrix

import keras.backend as K
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn.metrics


# Threshold probability vector to binary

def threshold(y_pred, thresh):
    return (y_pred >= thresh) * 1

# Ouput the ROC

def ROC(y_pred, y_true, plot = True):
    pred_sens = []
    pred_spec = []
    for thresh in range(0, 11):
        thresh /= 10
        sens = sens_value(y_pred, y_true, thresh)
        spec = spec_value(y_pred, y_true, thresh)
        pred_sens.append(sens)
        pred_spec.append(spec)

        print("Threshold: {:.1f}, Sensitivity: {:.3f}, Specificity: {:.3f}".format(thresh, sens, spec))
   
    # Calculate AUROC
    auroc = sklearn.metrics.roc_auc_score(y_true[:, 1], y_pred[:, 1])
    print("\nAUROC of {:.3f}".format(auroc))
    
    if plot:
                
            ## Plot figure

        x = np.linspace(0, 1, 50)

        fig, ax = plt.subplots()

        # Change the x to (1-spec) for the ROC curve
        roc, = ax.step(x = 1 - np.array(pred_spec), y = np.array(pred_sens), 
		    where = 'post', label = 'ROC curve with AUROC = {:.3f}'.format(auroc))
        equality, = ax.plot(x, x, dashes = [6, 2], label = 'Classification due to chance with AUROC = 0.5')
	    
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('Receiver Operating Charcteristics (ROC) Curve')
	    
        ax.legend()
        plt.show()


# Ouput binary confusion matrix

def confusion_matrix(y_pred, y_true, thresh = 0.5):
    # Threshold to turn probability vector to binary
    y_pred = threshold(y_pred, thresh)
    
    # Taking the True node from the binary output rather than both
    y_pred = y_pred[:, 1]
    y_true = y_true[:, 1]

    not_true = 1 - y_true
    not_pred = 1 - y_pred

    FP = K.sum(not_true * y_pred).eval(session = tf.Session())
    TN = K.sum(not_true * not_pred).eval(session = tf.Session())

    FN = K.sum(y_true * not_pred).eval(session = tf.Session())
    TP = K.sum(y_true * y_pred).eval(session = tf.Session())
    
    print('''
       BINARY CONFUSION MATRIX
            THRESHOLD: {}
	      +-----------------+
    	      |    predicted    |
    +---------+-----------------+
    |actually |True	|False	|
    |---------|---------|-------|
    |True     |{}	|{}	|
    |False    |{}	|{}	|
    +---------+---------+-------+
    '''.format(thresh, TP, FN, FP, TN))

def spec_value(y_pred, y_true, thresh = 0.5):

    """
    specificity:
    y_pred = matrix of labels that are predicted to be true
    y_true = labels that are true by the GT
    not_true = labels that are not true by the GT
    not_pred = labels that are predicted to be not true
    return spec
    """

    # Threshold to turn probability vector to binary
    y_pred = threshold(y_pred, thresh)

    # Taking the True node from the binary output rather than both
    y_pred = y_pred[:, 1]
    y_true = y_true[:, 1]
  
    not_true = 1 - y_true
    not_pred = 1 - y_pred

    FP = K.sum(not_true * y_pred).eval(session = tf.Session())
    TN = K.sum(not_true * not_pred).eval(session = tf.Session())

    spec = TN / (TN + FP)

    return spec


def sens_value(y_pred, y_true, thresh = 0.5):

    """
    sensitivity:
    y_pred = matrix of labels that are predicted to be true
    y_true = labels that are true by the GT
    not_true = labels that are not true by the GT
    not_pred = labels that are predicted to be not true
    return sens
    """

    # Threshold to turn probability vector to binary
    y_pred = threshold(y_pred, thresh)
    
    # Taking the True node from the binary output rather than both
    y_pred = y_pred[:, 1]
    y_true = y_true[:, 1]
    
    # Subtracting the y_true and y_pred from 1 gives the complement 
    not_true = 1 - y_true
    not_pred = 1 - y_pred

    FN = K.sum(y_true * not_pred).eval(session = tf.Session())
    TP = K.sum(y_true * y_pred).eval(session = tf.Session())

    sens = TP / (TP + FN)

    return sens

# Tensor metrics

def spec(y_pred, y_true):

    """
    specificity:
    y_pred = matrix of labels that are predicted to be true
    y_true = labels that are true by the GT
    not_true = labels that are not true by the GT
    not_pred = labels that are predicted to be not true
    return spec
    """
    # Subtracting the y_true and y_pred from 1 gives the complement 
    not_true = 1 - y_true
    not_pred = 1 - y_pred
	
    # Calculating FP and TN
    FP = K.sum(K.round(not_true * y_pred))
    TN = K.sum(K.round(not_true * not_pred))

    spec = TN / (TN + FP + K.epsilon())
    
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
    
    # Subtracting the y_true and y_pred from 1 gives the complement 
    not_true = 1 - y_true
    not_pred = 1 - y_pred
	
    # Calculating FN and TP
    FN = K.sum(K.round(y_true * not_pred))
    TP = K.sum(K.round(y_true * y_pred))

    sens = TP / (TP + FN + K.epsilon())    

    return sens 
