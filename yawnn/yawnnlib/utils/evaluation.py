import numpy as np

def evaluate(ground : np.ndarray, prediction : np.ndarray):
    """ Evaluate prediction. """
    precision = calcPrecision(ground, prediction)
    recall = calcRecall(ground, prediction)
    f1 = calcF1(recall, precision)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")

def calcPrecision(ground : np.ndarray, prediction : np.ndarray) -> float:
    """ Calculate precision. """
    if len(prediction) == 0:
        return 0.0
    return float(np.sum(ground & prediction) / np.sum(ground == prediction))

def calcRecall(ground : np.ndarray, prediction : np.ndarray) -> float:
    """ Calculate recall. """
    if len(ground) == 0:
        return 0.0
    return float(np.sum(ground & prediction) / np.sum(ground))

def calcF1(recall : float, precision : float) -> float:
    """ Calculate F1. """
    return 2 * (precision * recall) / (precision + recall)