'''
found at:
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
'''

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

print('starting.')
X, y = load_iris(return_X_y=True)
clf = LogisticRegression(random_state=0, max_iter=1000).fit(X, y)
pred1 = clf.predict(X[:2, :])
print(f'pred1: {pred1} ')
pred2 = clf.predict_proba(X[:2, :])
print(f'pred2: {pred2}')
print(clf.score(X, y))

print(data)


