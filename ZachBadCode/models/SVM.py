'''
https://scikit-learn.org/stable/modules/svm.html#classification
'''

from sklearn import svm

X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC()
clf.fit(X, y)
pred = clf.predict([[2., 2.]])
print(pred)


def fake_function():
    fake_obj = 'asdfasdf'
    return fake_obj