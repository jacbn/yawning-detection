import numpy as np

def _sqDist(x, y):
    d = x - y
    return np.dot(d, d)

def classify(point, data, classifications, k=3):
    ordered = sorted(zip(data, classifications), key=lambda z: _sqDist(point, np.array(z[0])))
    return max(l := list(map(lambda x: x[1], ordered[:k])), key=lambda x: l.count(x))

def classifyMultiple(points, data, classifications, k=3):
    return np.array([classify(p, data, classifications, k) for p in points])

if __name__ == "__main__":
    data = [[1, 1, 1, 1, 1, 1], [1, 1, 2, 3, 1, 1], [2, 1, 3, 1, 2, 1], [2, 3, 3, 3, 1, 2], [3, 2, 1, 1, 2, 3], [-5, -5, -5, -5, -5, -5], [-5, -3, -2, -5, -4, -1], [-5, -5, -2, -4, -3, -4]]
    classifications = [1, 1, 1, 1, 1, 0, 0, 0]
    assert classify([1, 0, 2, 1, 3, 1], data, classifications) == 1
    assert classify([0, 0, 5, 3, 2, 1], data, classifications) == 1
    assert classify([10, 10, 5, 2, 3, 1], data, classifications) == 1
    assert classify([1, -5, -5, -5, -4, -3], data, classifications) == 0
    print("All tests passed")