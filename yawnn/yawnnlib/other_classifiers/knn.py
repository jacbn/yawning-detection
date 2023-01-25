import numpy as np

def _quickSelect(l : list, k : int, acc=[]):
    if k == 0: return acc
    if len(l) < k: raise ValueError(f"Can't select {k} elements from a list of length {len(l)}!")
    
    pivot = l[np.random.randint(0, len(l)-1)]
    
    left, middle, right = [], [], []
    for el in l:
        left.append(el) if el < pivot else middle.append(el) if el == pivot else right.append(el)
    
    left += middle[1:]
    right += [middle[0]]

    if len(right) > k:
        # if enough elements in the right list that we don't need to recurse left
        return _quickSelect(right, k, acc)
    else:
        # if the length of the right list is not enough to satisfy k, then all of them must be in there, then recurse left with the rest
        return _quickSelect(left, k-len(right), acc + right)

# for each d in data, calculate the distance between d and point. return the most common classification of the k nearest neighbors
def classify(point, data, classifications, k=3):
    data, point, classifications = np.array(data), np.array(point), list(classifications)
    vec2 = np.sum(np.square(data - point), axis=1) # a vector of square magnitudes from each point in data to the given point
    
    vec2List = list(-vec2) # negative as quickSelect finds the max; list for compatibility
    maxEls = _quickSelect(vec2List, k)
    
    maxKIndices = list(map(lambda y: y[0], sorted([(i, el) for i, el in enumerate(vec2List) if el in maxEls], key=lambda x: x[1])[:k]))
    maxClassifications = list(np.array(classifications)[maxKIndices])

    return max(maxClassifications, key=lambda x: maxClassifications.count(x))

    # old method -- slow, sorted entire list:
    # ordered = sorted(zip(data, classifications), key=lambda z: _sqDist(point, np.array(z[0]))) # returns a list of (data, classification) tuples, sorted by distance from point
    # return max(l := list(map(lambda x: x[1], ordered[:k])), key=lambda x: l.count(x)) # returns the most common classification of the first k elements


def classifyMultiple(points, data, classifications, k=3):
    l = []
    for p in points:
        l.append(classify(p, data, classifications, k))
        print(f"Classifying points: {int(100 * len(l) / len(points))}%", end="\r" if len(l) != len(points) else "\n")
    return np.array(l)

if __name__ == "__main__":
    data = [[1, 1, 1, 1, 1, 1], [1, 1, 2, 3, 1, 1], [2, 1, 3, 1, 2, 1], [2, 3, 3, 3, 1, 2], [3, 2, 1, 1, 2, 3], [-5, -5, -5, -5, -5, -5], [-5, -3, -2, -5, -4, -1], [-5, -5, -2, -4, -3, -4]]
    classifications = [1, 1, 1, 1, 1, 0, 0, 0]
    assert classify([1, 0, 2, 1, 3, 1], data, classifications) == 1
    assert classify([0, 0, 5, 3, 2, 1], data, classifications) == 1
    assert classify([10, 10, 5, 2, 3, 1], data, classifications) == 1
    assert classify([1, -5, -5, -5, -4, -3], data, classifications) == 0
    print("All tests passed")