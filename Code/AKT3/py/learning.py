import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score,cross_validate
from rdkit import Chem
from mordred import Calculator, descriptors
from sklearn.model_selection import cross_val_score,LeaveOneOut
from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing
import math
from sklearn import metrics
from rdkit.Chem import MACCSkeys
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
name=[svm.SVC,KNeighborsClassifier,GaussianNB,MLPClassifier,XGBClassifier,RandomForestClassifier]
file_name=['estate','extended','graph','klekota-roth','maccs','pubchem','descriptor']
def QSAR(model_name,file_name):
    df = pd.read_csv('../Descriptor result/'+file_name+'_sample.csv')
    X=df.iloc[0:,1:-1]
    y = df.iloc[0:, -1]

    clf =model_name()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    clf.fit(X=X_train, y=y_train)
    y_pred=clf.predict(X_test)
    x_pred=clf.predict(X_train)
    matthews_train=matthews_corrcoef(y_train,x_pred)
    accuracy_train=accuracy_score(y_train,x_pred)
    Auc_train=roc_auc_score(y_train,x_pred)
    f1_train=f1_score(y_train,x_pred)
    precision_train=precision_score(y_train,x_pred)
    recall_train=recall_score(y_train,x_pred)
    BA_train=(precision_train+recall_train)/2
    metrics_train = confusion_matrix(y_train,x_pred)
    matthews_test=matthews_corrcoef(y_test, y_pred)
    accuracy_test=accuracy_score(y_test, y_pred)
    Auc_test=roc_auc_score(y_test, y_pred)
    f1_test=f1_score(y_test, y_pred)
    precision_test=precision_score(y_test, y_pred)
    recall_test=recall_score(y_test, y_pred)
    metrics_test = confusion_matrix(y_test, y_pred)
    BA_test=(precision_test+recall_test)/2
    accuracy_10 = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy').mean()
    auc_10 = cross_val_score(clf, X_train, y_train, cv=10, scoring='roc_auc').mean()
    f1_10 = cross_val_score(clf, X_train, y_train, cv=10, scoring='f1').mean()
    precision_10=cross_val_score(clf, X_train, y_train, cv=10, scoring='precision').mean()
    recall_10=cross_val_score(clf, X_train, y_train, cv=10, scoring='recall').mean()
    dic = {'matthews_train':matthews_train, 'accuracy_train': accuracy_train, 'Auc_train':Auc_train, 'f1_train': f1_train, 'precision_train': precision_train,'recall_train':recall_train,'metrics_train':metrics_train,'BA_train':BA_train,
           'matthews_test': matthews_test, 'accuracy_test': accuracy_test, 'Auc_test': Auc_test,'f1_test': f1_test, 'precision_test': precision_test, 'recall_test': recall_test, 'metrics_test': metrics_test,'BA_test':BA_test,
           'accuracy_10':accuracy_10,'precision_10':precision_10,'recall_10':recall_10,'auc_10':auc_10,'f1_10':f1_10}
    return dic

for i in name:
    result_final = pd.DataFrame()
    for j in file_name:
      result=QSAR(i,j)
      print(j)
      result_final=pd.concat([result_final,pd.DataFrame([result],index=[j])])
 #dic={str(i.__name__):result}
    print(result_final)
    result_final.to_csv('../learning result/'+i.__name__+'.csv')
