from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import accuracy_score

np.random.seed(17)
r = 0

for M in range(10000):
    X = np.random.normal(loc=0, size=(20, 2))
    y = np.array([0]*10 + [1]*10)
    model = LogisticRegression(penalty='none')
    model.fit(X, y)
    y_pred = model.predict(X)
    if accuracy_score(y, y_pred) == 1:
        r += 1

print(r/10000)



