from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
'''
Exercise: Load the MNIST data and split it into a training set, a validation 
set, and a test set (e.g., use 50,000 instances for training, 10,000 for 
validation, and 10,000 for testing).
'''
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

mnist.target = mnist.target.astype(np.uint8)

#split data into X and y
X = mnist['data']
y = mnist['target']

#validation and test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=10000, random_state=42)

#train and test split
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=10000, random_state=42)

#models obj   
rfc = RandomForestClassifier(n_estimators=100,random_state=42)
extrtree = ExtraTreesClassifier(n_estimators=100,random_state=42)
svm = LinearSVC(max_iter=100, tol=20, random_state=42)

#list of models
estimators = [rfc,extrtree,svm]

#traing models in estimator using train set
for estimator in estimators:
    print('training model:', estimator)
    estimator.fit(X_train,y_train)

#evaluation    
score = [estimator.score(X_train_val,y_train_val) for estimator in estimators]


#list of sets of models for ensemble to make votingclassifier
estimator = [('random',rfc),
             ('svm',svm),
             ('extra',extrtree)]

#voting classifier obj
voting = VotingClassifier(estimator)

#training voting classifier
voting.fit(X_train,y_train)

#evaluation of validation set
voting.score(X_val,y_val)

#for each estimator in voting classifier evalute on validation set
[estimator.score(X_val,y_val) for estimator in voting.estimators_]

voting.set_params(svm=None) #change the param value of svm to none

voting.estimators_

#or deleting svm classifier as its outperforms and affect the votiong model
del voting.estimators_[1]

voting.score(X_val, y_val)

voting.voting = 'soft'

voting.voting = 'hard'
    
#test set

[estimator.score(X_test,y_test) for estimator in voting.estimators_]

'''
Exercise: Run the individual classifiers from the previous exercise to make 
predictions on the validation set, and create a new training set with the 
resulting predictions: each training instance is a vector containing the set 
of predictions from all your classifiers for an image, and the target is the 
image's class. Train a classifier on this new training set.
'''
#making empty array for estimator prediction as datatype float32
X_val_pred = np.empty((len(X_val), len(estimators)), dtype = np.float32)


for index, estimator in enumerate(estimators):
    X_val_pred[:,index] =  estimator.predict(X_val)

#predicted array
X_val_pred

#blender model of random forest classifier (oob_score is out of bag which use
#holded out data which was left while training)
rnd_forest_blender = RandomForestClassifier(n_estimators = 200, oob_score = True
                                            , random_state = 42)

#training blender
rnd_forest_blender.fit(X_val_pred,y_val)

#oob_score
rnd_forest_blender.oob_score_


#test set evaluation
X_test_pred = np.empty((len(X_test), len(estimators)), dtype= np.float32)

for index, estimator in enumerate(estimators):
  X_test_pred[:,index] = estimator.predict(X_test)


#blender and classifier forms a stacking ensemble
#prediction
y_pred = rnd_forest_blender.predict(X_test_pred)

#accuracy score of rnd blender
accuracy_score(y_test, y_pred)

#blender model of extra tree classifier
extra_tree_blender = ExtraTreesClassifier(n_estimators=200,oob_score=True, 
                                          random_state=42,bootstrap=True)

extra_tree_blender.fit(X_val_pred,y_val)

#predicition
tree_pred = extra_tree_blender.predict(X_test_pred)

#accuracy score of extra tree blender
accuracy_score(y_test,tree_pred)
