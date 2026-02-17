import numpy as np
import pandas as pd

from ltcn.LTCN import LTCN

from sklearn import datasets
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

iris = datasets.load_iris(as_frame=True)

Y = pd.get_dummies(iris.target).values
X = iris.data.values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skf.get_n_splits(X, Y)

errors = []
for train_index, test_index in skf.split(X, iris.target):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    model = LTCN(T=5, phi=0.9, method="ridge", function="sigmoid", alpha=1.0E-4)
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)
    kappa = cohen_kappa_score(np.argmax(Y_test, axis=1), np.argmax(Y_pred, axis=1))
    errors.append(kappa)

print(sum(errors) / len(errors))