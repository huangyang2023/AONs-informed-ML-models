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
file_name=['estate','graph','klekota-roth','maccs','pubchem','descriptor']
def output(file_name):
    df = pd.read_csv('../Descriptor result/'+file_name)
    X=df.iloc[0:,1:-1]
    y = df.iloc[0:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    pd.concat([pd.DataFrame(X_train),pd.DataFrame(y_train)], axis=1).to_csv('../Descriptor result/descriptor_train.csv')
    pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test)], axis=1).to_csv('../Descriptor result/descriptor_test.csv')
output('descriptor1.csv')