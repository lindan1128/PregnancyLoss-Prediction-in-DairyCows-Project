__author__ = 'LIN Dan'

from keras import backend as K
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def specificity(y_true, y_pred):
    """
    param:
    y_pred - Predicted labels
    y_true - True labels 
    Returns:
    Specificity score
    """
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fp = K.sum(neg_y_true * y_pred)
    tn = K.sum(neg_y_true * neg_y_pred)
    specificity = tn / (tn + fp + K.epsilon())
    return specificity

def f1_metric(y_true, y_pred):
    """
    param:
    y_pred - Predicted labels
    y_true - True labels
    Returns:
    F1 score
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def convert_y(y):
    """
    param:
    y - labels
    Returns:
    one-hot encoded labels
    """
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False, categories="auto")
    y = label_encoder.fit_transform(y)
    y = y.reshape(len(y), 1)
    y = onehot_encoder.fit_transform(y)
    return y
