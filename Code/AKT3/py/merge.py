import pandas as pd
a=pd.read_csv('../Descriptor result/graph_sample.csv',index_col=0).iloc[:,0:-1]
b=pd.read_csv('../Descriptor result/klekota-roth_sample.csv',index_col=0).iloc[:,0:-1]
c=pd.read_csv('../Descriptor result/maccs_sample.csv',index_col=0).iloc[:,0:-1]
d=pd.read_csv('../Descriptor result/pubchem_sample.csv',index_col=0).iloc[:,0:-1]
e=pd.read_csv('../Descriptor result/descriptor_sample.csv',index_col=0).iloc[:,0:-1]
h=pd.read_csv('../Descriptor result/estate_sample.csv',index_col=0)
k=pd.concat([a,b],axis=1,join="inner")
k=pd.concat([k,c],axis=1,join="inner")
k=pd.concat([k,d],axis=1,join="inner")
k=pd.concat([k,e],axis=1,join="inner")
k=pd.concat([k,h],axis=1,join="inner")
print(k)
k.to_csv('../Descriptor result/descriptor1.csv')