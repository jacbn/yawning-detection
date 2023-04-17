import numpy as np

def evaluate(ground : np.ndarray, prediction : np.ndarray):
    """ Evaluate a prediction given the ground truth and the predicted values. """
    accuracy = calcAccuracy(ground, prediction)
    precision = calcPrecision(ground, prediction)
    recall = calcRecall(ground, prediction)
    f1 = calcF1(recall, precision)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")
    return accuracy, precision, recall, f1

def calcAccuracy(ground : np.ndarray, prediction : np.ndarray) -> float:
    """ Calculate accuracy, TP+TN/(TP+TN+FP+FN). """
    if len(ground) == 0:
        return 0.0
    return float(np.sum(ground == prediction) / len(ground))

def calcPrecision(ground : np.ndarray, prediction : np.ndarray) -> float:
    """ Calculate precision, TP/(TP+FP). """
    if np.sum(prediction) == 0:
        return 0.0
    return float(np.sum(ground & prediction) / np.sum(prediction))

def calcRecall(ground : np.ndarray, prediction : np.ndarray) -> float:
    """ Calculate recall, TP/(TP+FN) """
    if np.sum(ground) == 0:
        return 0.0
    return float(np.sum(ground & prediction) / np.sum(ground))

def calcF1(recall : float, precision : float) -> float:
    """ Calculate F1, the harmonic mean of precision and recall. """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)
