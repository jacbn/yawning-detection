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
    for p in range(len(points)):
        l.append(classify(p, data, classifications, k=k, tree=tree))
        if p % 100 == 0:
            print(f"Classifying points: {int(100 * len(l) / len(points))}%", end="\r" if len(l) != len(points) else "\n")
    return np.array(l)
