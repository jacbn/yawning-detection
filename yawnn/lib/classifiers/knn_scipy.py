from scipy import spatial
import numpy as np

def classify(point, data, classifications, k=3, tree=None):
    assert type(data) == np.ndarray
    if tree is None:
        tree = spatial.KDTree(data)
    query = tree.query(point, k=k)[1].astype(int)
    nearest = data[query]
    ind = []
    for row in nearest:
        ind.append(np.where(np.all(row == data, axis=1))[0][0])
    classInd = np.array(classifications)[ind]
    return max(classInd, key=lambda x: classInd.tolist().count(x))

def classifyMultiple(points, data, classifications, k=3):
    tree = spatial.KDTree(data)
    l = []
    for p in points:
        l.append(classify(p, data, classifications, k=k, tree=tree))
        print(f"Classifying points: {int(100 * len(l) / len(points))}%", end="\r" if len(l) != len(points) else "\n")
    return np.array(l)

if __name__ == "__main__":
    data = np.array([(-5, -5, -5, -5, -5, -5), (1, 1, 1, 1, 1, 1), (1, 1, 2, 3, 1, 1), (2, 1, 3, 1, 2, 1), (2, 3, 3, 3, 1, 2), (3, 2, 1, 1, 2, 3) , (-5, -3, -2, -5, -4, -1), (-5, -5, -2, -4, -3, -4)])
    classifications = [1, 1, 1, 1, 1, 0, 0, 0]
    assert classify([1, 0, 2, 1, 3, 1], data, classifications) == 1
    assert classify([0, 0, 5, 3, 2, 1], data, classifications) == 1
    assert classify([10, 10, 5, 2, 3, 1], data, classifications) == 1
    assert classify([1, -5, -5, -5, -4, -3], data, classifications) == 0
    print("All tests passed")
