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

from sklearn.inspection import permutation_importance
name=[svm.SVC,KNeighborsClassifier,GaussianNB,MLPClassifier,XGBClassifier,RandomForestClassifier]
file_name=['estate','extended','graph','klekota-roth','maccs','pubchem','descriptor']
def importance(model_name,file_name):
    df = pd.read_csv('../Descriptor result/' + file_name + '_sample.csv')
    X = df.iloc[0:, 1:-1]
    print(X.columns.tolist())
    y = df.iloc[0:, -1]

    clf = model_name()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    clf.fit(X=X_train, y=y_train)
    imps = permutation_importance(clf, X_test, y_test)

    im = pd.DataFrame(imps.importances_mean,index=X.columns.tolist())
    print(im)

    return im


def singe():
    cc = importance(RandomForestClassifier, 'klekota-roth')
    writer = pd.ExcelWriter(r"importance.xlsx", mode="a", engine="openpyxl")
    cc.to_excel(writer, index=True, sheet_name='name')
    writer.save()
    writer.close()
def multiple():
    for i in range(len(file_name)):
        for j in range(len(name)):
            cc=importance(name[j], file_name[i])
            writer = pd.ExcelWriter(r"importance.xlsx", mode="a", engine="openpyxl")
            cc.to_excel(writer, index=True, sheet_name=name[j].__name__+'+'+file_name[i])
            writer.save()
            writer.close()
#multiple()
singe()