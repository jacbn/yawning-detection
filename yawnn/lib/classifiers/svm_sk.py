from sklearn import svm

def fitSVM(X, y):
    clf = svm.SVC(kernel='rbf')
    clf.fit(X, y)
    return clf
    
# the best hyperplane is the one that leaves the maximum margin between the two classes
# that is, the distance between the closest point of each class is maximized

# whichever line gives us the greatest margin is the best hyperplane
# however, the hyperplane need not be linear

# define hyperplane equation to be g(X) = W^T * X + b   (capital indicates a vector)
# in an ideal hyperplane, all points X in class 1 will satisfy g(X) >= 1, and all points X in class 2 will satisfy g(X) <= -1

# we know that the distance between a hyperplane and the closest point is z = |g(X)|/||W|| = 1/||W||  (why??)
# the total margin between the two closest points and the hyperplane, then, is 2/||W||
# minimising the term 2/||W|| will maximise the separability of the two classes

# how do we minimise W: Karush-Kuhn-Tucker conditions.
# main conditions are:
# - W = sum_i^N(lambda_i * y_i * X_i)
# - sum_i^N(lambda_i * y_i) = 0
