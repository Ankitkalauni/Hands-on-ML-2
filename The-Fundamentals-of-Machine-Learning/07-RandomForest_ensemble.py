from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier


mnist = fetch_openml('mnist_784', version=1, as_frame=False)

mnist.target = mnist.target.astype(np.uint8)

X = mnist['data']
y = mnist['target']

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=10000, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=10000, random_state=42)
    
rfc = RandomForestClassifier(n_estimators=100,random_state=42)
extrtree = ExtraTreesClassifier(n_estimators=100,random_state=42)
svm = LinearSVC(max_iter=100, tol=20, random_state=42)


estimators = [rfc,extrtree,svm]

for estimator in estimators:
    print('training model:', estimator)
    estimator.fit(X_train,y_train)
    
score = [estimator.score(X_train_val,y_train_val) for estimator in estimators]



estimator = [('random',rfc),
             ('svm',svm),
             ('extra',extrtree)]

voting = VotingClassifier(estimator)

voting.fit(X_train,y_train)

voting.score(X_val,y_val)
    
[estimator.score(X_val,y_val) for estimator in voting.estimators_]

voting.set_params(svm=None) #change the param value of svm to none

voting.estimators_

del voting.estimators_[1]

voting.score(X_val, y_val)

voting.voting = 'soft'

voting.voting = 'hard'
    
#test set

[estimator.score(X_test,y_test) for estimator in voting.estimators_]
