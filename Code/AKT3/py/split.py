import pandas as pd
import numpy as np
from rdkit import Chem
from mordred import Calculator, descriptors
data=pd.read_csv('../Descriptor result/sample.csv')
print(data)
def colum(colum_name):
  f=open('../fingerprinter/'+colum_name+'.txt')
  name=[]
  for line in f:
    name.extend(line.split(','))
    name.append('outcome')
  return name
fingertypes= ['graph','maccs','pubchem','estate','klekota-roth','extended']
file_name=['estate','extended','graph','klekota-roth','maccs','pubchem','descriptor']
for i in range(len(fingertypes)):
    data2=data.loc[:,colum(fingertypes[i])]
    print(data2)
    data2.to_csv('../Descriptor result/'+fingertypes[i]+'_sample.csv')
data2=data.loc[:,:'graph0']

data2.rename(columns = {'graph0':'outcome'},inplace=True)

data2['outcome']=data.iloc[:,-1]

data2=data2.iloc[:,1:]
data2.to_csv('../Descriptor result/descriptor_sample.csv')
print(data2)
