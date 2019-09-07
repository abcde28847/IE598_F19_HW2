#!/usr/bin/env python
# coding: utf-8

# In[308]:


# immport all the package we need,such as knn package and decision tree package
from sklearn import datasets
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[309]:


# read the Treasury Squeeze test.xls file into the storage
TS = pd.read_csv("Treasury Squeeze test.xls",sep = ",")


# In[310]:


# choose two columns as the X dataset and the last column as the y dataset.
X = TS.iloc[1:,7:9]
y = TS.iloc[1:,-1]
# we use print to make sure whether we read a correct file.
print(X.shape,y.shape)


# In[311]:


# Divide the X and y dataset into X_train, X_test,y_train and y_test, and the retio is train:test = 3:1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)


# In[312]:


# use the for-loop to calculate a best k which is used by KNN
k_range = range(1,26)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))


# In[313]:


sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


# In[314]:


# find the best score and we can get the best way to use KNN
max_score = scores.index(np.max(scores))+1


# In[315]:


# Found the following function from the book ,we can use it to plot some graph.
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
# setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=colors[idx], marker=markers[idx], label=cl, edgecolor='black')
    # highlight test samples
    if test_idx:
    # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', edgecolor='black', alpha=1.0, linewidth=1, marker='o', s=100, label='test set')


# In[316]:


# use the best score to plot the plot the graph
knn_new = KNeighborsClassifier(n_neighbors=max_score)
X_combined=np.vstack((X_train_std, X_test_std))
X_combined_std = sc.transform(X_combined)
y_combined = np.hstack((y_train,y_test))
# A new KNN to plot
knn_new.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=knn_new, test_idx=range(105,150))
plt.xlabel('petal length [near_minus_next]')
plt.ylabel('petal width [ctd_last_first]')
plt.legend(loc='upper left')
plt.show()


# In[317]:


# Use the decision tree to classifier 
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='gini',max_depth=4,random_state=1)
tree.fit(X_train, y_train)
# A new combination dataset
X_combined_DTnew = np.vstack((X_train, X_test))
y_combined_DTnew = np.hstack((y_train, y_test))
plot_decision_regions(X_combined_DTnew,y_combined_DTnew,classifier=tree,test_idx=range(105, 150))
plt.xlabel('near_minus_next]')
plt.ylabel('ctd_last_first')
plt.legend(loc='upper left')
plt.show()


# In[318]:


print("My name is {Wanrong Cai}")
print("My NetID is: {wanrong2}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")


# In[ ]:




