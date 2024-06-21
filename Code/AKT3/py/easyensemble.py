import pandas as pd
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from imblearn.under_sampling import NearMiss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

def easyensemble():
    data=pd.read_csv('../Descriptor result/descriptor.csv')
    X=data.iloc[1:,1:-1]
    y=data.iloc[1:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=0)
    cc = NearMiss(version=3)
    X_resampled, y_resampled = cc.fit_resample(X,y)
    pd.concat([X_resampled,y_resampled],axis=1).to_csv('../Descriptor result/sample.csv')
    print(pd.concat([X_resampled,y_resampled],axis=1))
easyensemble()
