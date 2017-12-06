 #!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 13:43:04 2017

@author: Alex
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn import metrics


def ConvCharVec2NumbVec(CharVec):
    # this function will take a vector of characters and convert it to a correspon
    # ding int vector. A range of int value will be determined by the length of 
    # unique function value of given character vector.
    
    IntVec = np.zeros(CharVec.shape)
    CharVecUnique = np.unique(CharVec)
    
    # now, create the random vectors to assign values
    FirstRand = np.random.uniform(0,10,1)[0] # mean
    SecondRand = np.random.uniform(0.5,3,1)[0] # increment
    RandVal = np.zeros(CharVecUnique.shape)
    
    for i in range(0,CharVecUnique.shape[0]):
        RandVal[i] = FirstRand + (i*SecondRand)
        
    for i in range(0,CharVec.shape[0]):
        ThisChar = CharVec[i]
        CorrespInd = np.where(CharVecUnique==ThisChar)[0][0]
        IntVec[i] = RandVal[CorrespInd]
              
    return IntVec

# first, we need to take a look at the DataFrame if there is any NaN value
df = pd.read_csv('mushrooms.csv')

X = df.iloc[:,1:] # feature values are characters
y = df.iloc[:,0] # labels. either 0 or 1

XVal = X.values
yVal = (y.values == 'e')*1 # 1 if it is edible, 0 otherwise
       
FVerXVal = np.zeros(XVal.shape)

# convert XVal to float version
for i in range(0,XVal.shape[1]):
    FVerXVal[:,i] = ConvCharVec2NumbVec(XVal[:,i])
    
# plot the correlation of data
sns.heatmap(np.corrcoef(np.transpose(FVerXVal)), square=True, cmap="RdYlGn", xticklabels=list(X), yticklabels=list(X))
plt.show()

# split into train and test data
X_train, X_test, y_train, y_test = train_test_split(FVerXVal, yVal)

# data can be scaled depending on their minimum or maximum values
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# exponent data for L1 regularization parameter
B = np.arange(-10,10).astype(float)
Power = np.zeros(B.shape)
for i in range(0,len(B)):
    Power[i] = 10**B[i]

TrainResult = np.zeros((len(B)))
TestResult = np.zeros((len(B)))

# logistic regression
for i in range(0,len(Power)):
    clf = LogisticRegression(C=Power[i], penalty='l1').fit(X_train_scaled, y_train)
    Title = 'Logistic regression, C = {:.5f}'.format(Power[i])
    fpr, tpr, thresh = metrics.roc_curve(y_train, clf.predict(X_train_scaled))
    TrainResult[i] = metrics.auc(fpr, tpr)
    fpr, tpr, thresh = metrics.roc_curve(y_test, clf.predict(X_test_scaled))
    TestResult[i] = metrics.auc(fpr, tpr)

# plot the result; which C value I should use?
plt.plot(B,TrainResult)
plt.plot(B,TestResult)
plt.show()



















## decision tree
#clf = DecisionTreeClassifier(max_depth = 5).fit(X_train_scaled, y_train)
#
#print('Accuracy of Decision Tree classifier on training set: {:.2f}'
#     .format(clf.score(X_train_scaled, y_train)))
#print('Accuracy of Decision Tree classifier on test set: {:.2f}'
#     .format(clf.score(X_test_scaled, y_test)))

## random forest
#clf = RandomForestClassifier(max_features = 8, random_state = 0)
#clf.fit(X_train_scaled, y_train)
#
#print('Accuracy of RF classifier on training set: {:.2f}'
#     .format(clf.score(X_train_scaled, y_train)))
#print('Accuracy of RF classifier on test set: {:.2f}'
#     .format(clf.score(X_test_scaled, y_test)))

## gradient boosted
#clf = GradientBoostingClassifier(learning_rate = 0.01, max_depth = 2, random_state = 0)
#clf.fit(X_train_scaled, y_train)
#
#print('Accuracy of GBDT classifier on training set: {:.2f}'
#     .format(clf.score(X_train_scaled, y_train)))
#print('Accuracy of GBDT classifier on test set: {:.2f}\n'
#     .format(clf.score(X_test_scaled, y_test)))

# neural network
#for this_alpha in [0.01, 0.1, 1.0, 5.0]:
#    nnclf = MLPClassifier(hidden_layer_sizes = [1, 1], activation = 'tanh', solver='lbfgs', alpha = this_alpha).fit(X_train_scaled, y_train)
#    print('Accuracy of MLPClassifier on training set: {:.2f}\n'.format(nnclf.score(X_train_scaled, y_train)))
#    print('Accuracy of MLPClassifier on test set: {:.2f}\n'.format(nnclf.score(X_test_scaled, y_test)))

## SVM
#from sklearn.svm import SVC
#model = SVC().fit(X_train,y_train)
#SVMAUC = roc_auc_score(model.predict(X_test), y_test)
#
## plot them
#fig, ax = plt.subplots()
#
#pm, pc = plt.bar([0,1],[LogisticAUC,SVMAUC])
#pm.set_facecolor('r')
#pc.set_facecolor('g')
#ax.set_xticks([0,1])
#ax.set_xticklabels(['Logistic regression', 'SVM'])
#ax.set_ylim([0, 1])
#ax.set_ylabel('AUC')
#ax.set_title('AUC for different models')