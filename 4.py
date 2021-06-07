import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('BRCA_pam50.tsv', sep='\t', index_col=0)
df = df.loc[df['Subtype'].isin(['Luminal A', 'Luminal B'])]
X = df.iloc[:, :-1].to_numpy()
y = df['Subtype'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=17)
model = SVC(kernel='linear', class_weight='balanced')

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(balanced_accuracy_score(y_test, y_pred))

w = model.coef_[0]
genes = df.iloc[:,:-1].T
genes['coef'] = np.abs(w)
print(genes.sort_values(by='coef').iloc[-2:])

df2 = df[['BIRC5', 'BAG1', 'Subtype']]
X2 = df2.iloc[:, :-1].to_numpy()
y2 = df2['Subtype'].to_numpy()

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, stratify=y, random_state=17)
model = SVC(kernel='linear', class_weight='balanced')

model.fit(X2_train, y2_train)
y_pred = model.predict(X2_test)

print(balanced_accuracy_score(y2_test, y_pred))

model = LogisticRegression(class_weight='balanced', C=0.01, penalty='l1', solver='liblinear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(balanced_accuracy_score(y_test, y_pred))