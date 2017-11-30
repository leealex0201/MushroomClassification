#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 13:43:04 2017

@author: Alex
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

df = pd.read_csv('mushrooms.csv')
df2 = pd.get_dummies(df)

df3 = df2.sample(frac=0.08)

x = df3.iloc[:,2:]
y = df3.iloc[:,1]

pca = PCA(n_components=2).fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(pca, y, random_state=0)

from sklearn.linear_model import LogisticRegression

# logistic regression model
model = LogisticRegression()
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
LogisticAUC = roc_auc_score(y_predict, y_test)

# SVM
from sklearn.svm import SVC
model = SVC().fit(X_train,y_train)
SVMAUC = roc_auc_score(model.predict(X_test), y_test)

# plot them
fig, ax = plt.subplots()

pm, pc = plt.bar([0,1],[LogisticAUC,SVMAUC])
pm.set_facecolor('r')
pc.set_facecolor('g')
ax.set_xticks([0,1])
ax.set_xticklabels(['Logistic regression', 'SVM'])
ax.set_ylim([0, 1])
ax.set_ylabel('AUC')
ax.set_title('AUC for different models')