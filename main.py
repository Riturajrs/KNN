import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df1 = pd.read_csv('glass.csv')
X = df1[['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe']]
y = df1['Type']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.65)
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
scores = {}
scores_list = []
k1 = 0
acc = 0
for k in range(1,25):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    scores[k] = metrics.accuracy_score(y_test,y_pred)
    scores_list.append(scores[k])
    if scores[k] > acc:
        acc = scores[k]
        k1 = k
print(k1," ", acc)
plt.plot(range(1,25),scores_list)
plt.xlabel("Value of K for KNN")
plt.ylabel('Testing accuracy')
plt.show()
