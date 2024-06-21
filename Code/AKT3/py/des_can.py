import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors
import numpy as np
import time
from PyFingerprint.fingerprint import  get_fingerprints
#'standard', 'extended', 'graph', 'maccs', 'pubchem', 'estate', 'hybridization', 'lingo', 'klekota-roth', 'shortestpath', 'signature', 'substructure'
#'rdkit', 'morgan', 'rdk-maccs', 'topological-torsion', 'avalon', 'atom-pair'
#'fp2', 'fp3', 'fp4'




def colum(colum_name):
  f=open('../fingerprinter/'+colum_name+'.txt')
  name=[]
  for line in f:
    name.extend(line.split(','))
  return name




def fingprinter(data):
    fingertypes= ['graph','maccs','pubchem','estate','klekota-roth','extended']
    smlist = data['smile'].tolist()
    output = {}

    for f in fingertypes:
        output[f] = get_fingerprints(smlist, f)

    output_np = output.copy()
    df_empty=pd.DataFrame()
    for k, fps in output.items():
        output_np[k] = pd.DataFrame([fp.to_numpy() for fp in fps],columns=colum(k),index=data['name'])
        df_empty=pd.concat([df_empty,output_np[k]],axis=1)
    return df_empty
def des_gen(data):
 calc = Calculator(descriptors, ignore_3D=True)
 df=data['smile'].tolist()
 df_empty = pd.DataFrame()
 for i in range(len(df)):
   mol = Chem.MolFromSmiles(df[i])
   dfp = pd.DataFrame([calc(mol)])
   df_empty = pd.concat([df_empty, dfp], axis=0)
   print(i)
 df_empty.columns=calc.descriptors
 df_empty.index=data['name']
 rest = df_empty._get_numeric_data()
 return rest

def des_gen2(data):
    calc = Calculator(descriptors, ignore_3D=True)
    df = data['smile'].tolist()
    mols = [Chem.MolFromSmiles(smi) for smi in df]
    df2 = calc.pandas(mols,nproc=1)
    df2.columns=calc.descriptors
    rest = df2._get_numeric_data()
    rest.index=data['name']
    return rest
data=pd.read_csv('../Descriptor result/new.csv')

a=des_gen2(data)
b=fingprinter(data)
c=pd.concat([a,b],axis=1,join="inner")
s1=pd.DataFrame(data['outcome'].tolist(),index=data['name'],columns=['outcome'])
print(s1)
d= pd.concat([c, s1], axis=1)
print(d)
d.to_csv('../Descriptor result/descriptor.csv')






