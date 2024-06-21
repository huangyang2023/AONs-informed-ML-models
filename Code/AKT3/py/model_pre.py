import joblib
import pandas as pd
from sklearn import svm
from rdkit import Chem
from mordred import Calculator, descriptors
import numpy as np
from PyFingerprint.fingerprint import  get_fingerprints

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
def colum(colum_name):
  f=open('../fingerprinter/'+colum_name+'.txt')
  name=[]
  for line in f:
    name.extend(line.split(','))
  return name

def fingprinter(smile,type):
    fingertypes= type
    output = {}
    smlist = smile
    for f in fingertypes:
        output[f] = get_fingerprints(smlist, f)

    output_np = output.copy()

    for k, fps in output.items():
        output_np[k] = np.array([fp.to_numpy() for fp in fps])

    return pd.DataFrame(output_np[k])

def descriport(smile):
    mol = Chem.MolFromSmiles(smile)
    calc = Calculator(descriptors, ignore_3D=True)
    colu = pd.read_csv('/public/newtyf/running/fingerprinter/tt.csv')
    df = pd.DataFrame([calc(mol)], columns=colu.columns.tolist())
    im=pd.read_csv('../Descriptor result/descriptor_sample.csv',index_col=0).iloc[:,0:-1]
    idx=im.columns
    return df.loc[:,idx]

def using(model_name):
    smile=pd.read_csv('../prediction/smile.CSV')['Smile'].tolist()
    type=  pd.read_csv('../bestmodel/bestmodel.csv', index_col=None, header=0)['type'].tolist()
    for i in range(len(type)):
        new=[]
        new.append(type[i])
        model2 = joblib.load(filename='../bestmodel/' + type[i] + '_' + model_name.__name__ + '.model')
        if type[i]=='descriptor':
            result=[]
            for j in range(len(smile)):
                fingerprinter_ = np.array(descriport(smile[j]))
                try:
                    result.append(model2.predict( fingerprinter_))
                except:
                    result.append('na')
            return result
        else:
            fingerprinter_ = fingprinter(smile, new)
            model2.predict(fingerprinter_)
            return model2.predict(fingerprinter_)


data=pd.read_csv('../bestmodel/bestmodel.csv',index_col=None,header=0)
res = data.values.tolist()

for i in range(len(data)):
    result=using(eval(res[i][0]))
    pd.DataFrame(result).to_csv('../prediction/'+res[i][0]+res[i][1]+'.csv')
