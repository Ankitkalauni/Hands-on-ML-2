from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import normalize
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import zipfile
import numpy as np
import pandas as pd
import os


local = '/home/ankit/Downloads'

missing_values = ["n/a", "na", "--"]

train_link = os.path.join(local, 'titanic/train.csv')
test_link = os.path.join(local, 'titanic/test.csv')
gend1er_link = os.path.join(local, 'titanic/gender_submission.csv')

#dataframe import
train = pd.read_csv(train_link, na_values=missing_values)
test = pd.read_csv(test_link)
gender = pd.read_csv(gend1er_link)




def trans(data):
    #drop label first and then run this
    train = data

    train.dropna(subset=['Embarked'],inplace=True)

    train = train.reset_index(drop=True)

    #custom transformation
    class custom_Cabin(BaseEstimator, TransformerMixin):
        def __init__(self,data):
            self.data = data
        def fit(data):
            pass
        def transform(data):
            data.fillna('0', inplace=True) # replace nan with 0
            data_ar = data.values # into array
            data_cat = np.where(data_ar != '0',1,data_ar)
            data_int = data_cat.astype('int')
            cols = data.keys()[0]
            return pd.DataFrame(data_int,columns=[cols])
            
    cabin_custom = custom_Cabin.transform(train['Cabin'])
    
    # new feature
    train['relatives'] = train['SibSp'] + train['Parch']
    
    hot = OneHotEncoder(sparse=False)
    rod = OrdinalEncoder()
    
    train_cat = hot.fit_transform(train[['Sex']])
    sexcat = list(np.unique(train[['Sex']]))
    sex_df = pd.DataFrame(train_cat, columns=sexcat)
    sex_df = sex_df.add_prefix('sex_')
    
    train.dropna(subset=['Embarked'],inplace=True)
    embarked_cat = rod.fit_transform(train[['Embarked']])
    embarked_df = pd.DataFrame(embarked_cat, columns=['Embarked'])
    
    class custom_Cabin(BaseEstimator, TransformerMixin):
        def __init__(self,data):
            self.data = data
        def fit(data):
            pass
        def transform(data):
            data.fillna('0', inplace=True) # replace nan with 0
            data_ar = data.values # into array
            data_cat = np.where(data_ar != '0',1,data_ar)
            return data_cat.astype('int')
            
                
    cabin_custom = custom_Cabin.transform(train['Cabin'])
       
    cabin_custom = pd.DataFrame(cabin_custom, columns=['has_cabin'])
    train.drop(['Embarked'], axis=1, inplace=True)
    train = train.join([embarked_df,sex_df,cabin_custom])
    
    cont = train[['Age','Fare']]
    cont_cols = list(cont.columns)
    simp = SimpleImputer()
    cont = simp.fit_transform(cont)
    
    
    cont_nor = normalize(cont)
    cont_nor = pd.DataFrame(cont_nor, columns = cont_cols)
    cont_nor = cont_nor.add_prefix('norm_')
    
    end1 = pd.concat([train, cont_nor], axis=1)
    
    
    end1.drop(['SibSp', 'Parch', 'Ticket', 'Name','Age','Fare','sex_female','Sex', 'Cabin'],axis=1,inplace=True)
    end1.set_index(['PassengerId'], inplace=True)

    return end1



#calling function and modifying data    
train_mod = trans(train)
test_mod = trans(test)

train_label = train_mod['Survived']
train_mod.drop(['Survived'], axis=1, inplace=True)


train_mod.isnull().sum()

#parameter of the model
param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]

# =============================================================================
# #support vector machine
# from sklearn import svm
# lg = svm.SVC()
# 
# #Grid search cross validation
# from sklearn.model_selection import GridSearchCV
# grid = GridSearchCV(lg,param_grid, verbose=2, n_jobs=-1,cv=3)
# 
# grid.fit(train_mod,np.ravel(train_label))
# 
# print('Best estimator parameters: ', grid.best_estimator_)
# 
# pred = grid.predict(test_mod)
# 
# =============================================================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

randf = RandomForestClassifier()

para = {'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

rands =  RandomForestClassifier(randf, para, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

rands.fit(train_mod,np.ravel(train_label))

pred = rands.predict(test_mod)

from sklearn.metrics import confusion_matrix
print('confusion_matrix: ', confusion_matrix(pred,gender['Survived']))

from sklearn.metrics import accuracy_score
print('accuracy_score: ', accuracy_score(pred, gender['Survived']))
