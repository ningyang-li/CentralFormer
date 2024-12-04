#  -*- coding: utf-8 -*-

import numpy as np
from sklearn import metrics
import cv2


# color code,  format:(B, G, R)
map_table = [[0, 0, 0], 
           [128, 255, 128], [128, 128, 0], [64, 128, 128], [128, 255, 255], [0, 128, 255], 
           [255, 128, 192], [255, 128, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255], 
           [0, 255, 64], [0, 255, 255], [0, 128, 192], [128, 128, 192], [255, 0, 255], 
           [255, 128, 64], [0, 128, 128], [128, 128, 255], [255, 0, 128], [255, 128, 0]]


def compute_metrics(y_true_1hot, y_pred_1hot, n_category, pred_is_1hot=True):
    '''
    compute the metrics (recall,  overall accuracy,  average accuracy,  f1_score,  ...) from the predicted labels
    
    y_true_1hot:    true label (1-hot)
    y_pred_1hot:    predicted label (1-hot)
    n_category:     number of the category
    pred_is_1hot:   flag that if "y_pred_1hot" is the data_format of "1-hot"
    '''
    assert pred_is_1hot == True or pred_is_1hot == False
    assert n_category > 0
    assert pred_is_1hot and len(y_true_1hot.shape) == 2 and y_true_1hot.shape[-1] == n_category
    assert y_pred_1hot.shape == y_true_1hot.shape
    
    y_true = np.argmax(y_true_1hot, axis=1)
    if pred_is_1hot == True:
        y_pred = np.argmax(y_pred_1hot, axis=1)
    else:
        y_pred = y_pred_1hot
    class_names = []
    for i in range(n_category):
        class_names.append("c" + str(i + 1))
    
    return metrics.classification_report(y_true, y_pred, target_names=class_names, digits=4)


def confusion_matrix(y_true_1hot, y_pred_1hot, n_category, pred_is_1hot=True):
    '''
    get the confusion matrix
    
    y_true_1hot:    true label (1-hot)
    y_pred_1hot:    predicted label (1-hot)
    n_category:     number of the category
    pred_is_1hot:   flag that if "y_pred_1hot" is the data_format of "1-hot"
    '''
    assert pred_is_1hot == True or pred_is_1hot == False
    assert n_category > 0
    assert pred_is_1hot and len(y_true_1hot.shape) == 2 and y_true_1hot.shape[-1] == n_category
    assert y_pred_1hot.shape == y_true_1hot.shape
    
    y_true=np.argmax(y_true_1hot, axis=1)
    if pred_is_1hot == True:
        y_pred = np.argmax(y_pred_1hot, axis = 1)
    else:
        y_pred = y_pred_1hot
    labels = np.arange(n_category)
    
    return metrics.confusion_matrix(y_true, y_pred, labels=labels)

def kappa(confusion):
    '''
    get the kappa coefficient of classification results
    
    confusion: the confusion matrix got by the function [ConfusionMatrix]
    '''
    assert len(confusion.shape) == 2 and confusion.shape[0] == confusion.shape[1]
    
    N = np.sum(confusion)
    oa = np.sum(np.diag(confusion)) / N
    
    tp_fn = np.sum(confusion, axis=0)
    tp_fp = np.sum(confusion, axis=1)
    
    return (oa - np.sum(tp_fn * tp_fp) / np.power(N, 2)) / (1 - np.sum(tp_fn * tp_fp) / np.power(N, 2))


def get_predicted_map(pred, pos, row, col, is_GT=False):
    '''
    Get the predicted labels of all pixels in "pred",  where each element of "pred" is a
    scalar not a one-hot vector.
    if "pred" is the original output of network,  the operation "np.argmax(pred, axis=1)"
    is needed.
    
    pred:   the predicted labels of a batch or the test set. It isn't a one-hot vector, 
            the shape of it is (n_sp, )
    pos:    the position (row, col) of each sample in "pred",  which should be saved
            during the sampling of training,  test,  and validation sets.
    row:    the row of the original height of data set.
    col:    the column of the original width of data set.
    is_GT:  when you need to generate the ground-truth map of data set,  set "is_GT"
            is True. At this time,  "pred" should be all labeled pixels of data set
    '''
    assert len(pred.shape) == 1
    assert len(pos) == pred.shape[0]
    assert row > 0 and col > 0
    
    pred_map = np.zeros((row, col), np.int32)
    n = pos.shape[0]
    
    if is_GT == False:
        # when GT_model is False,  the labels of samples don't contain the background.
        pred += 1
    
    for i in range(n):
        pred_map[pos[i, 0], pos[i, 1]] = pred[i]
    
    return pred_map


def save_predicted_map(pred_map, path=""):
    '''
    Save the predicted classification map,  the "map_table" in the head of this file is used.
    
    pred_map:   an 2-d array got by the function [GetPredictMap],  which represents the labels of all or partial pixels of data set. 
    path:       the save path of the classification map,  e.g. "D:\\Save\\cls_map.png"
    '''
    assert len(pred_map.shape) == 2
    
    color = np.zeros((pred_map.shape[0], pred_map.shape[1], 3), dtype=np.int32)
    for i in range(pred_map.shape[0]):
        for j in range(pred_map.shape[1]):
            color[i, j, :] = map_table[pred_map[i, j]]
    color = color.astype(np.uint8)
    
    if path != "":
        if cv2.imwrite(path, color) == False:
            print("Failed to save predict map!")
        else:
            print("Success to save predict map!")
    else:
        #  cv2.namedWindow("color", cv2.WINDOW_NORMAL)
        cv2.imshow("color", color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        