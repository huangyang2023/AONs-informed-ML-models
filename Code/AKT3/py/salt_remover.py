import pandas as pd
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
import re
#读取数据，需要改一下路径，并保证excel的后缀名、文件内部需要有一列列名为smile
data=pd.read_excel(r'../Descriptor result/tox21.xlsx',sheet_name='Sheet1')
smis = data['smile'].tolist()
#这里是匹配的金属和铵根，添加只需加一个|并加上金属名称即可
findlink = re.compile(r'(Na\+|Hg\+|K\+|Mg\+|Al\+|Zn\+|Fe\+|Sn\+|Pb\+|Cu\+|NH4\+|Ag\+|Ca\+|Ba\+|OH-|Cl-|\.|Br-|Sb|Nd|Yb|Au|Ni)')
findlink2=re.compile(r'(C)')
new=[]
delet = []
for i in smis:
 link = re.findall(findlink, i)
 link2=re.findall(findlink2,i)
 if len(link) != 0 or len(link2)==0 :
  delet.append(i)
  # 输出删除化合物的名称
  print(link)
 else :
  new.append(i)
#输出删除化合物的数量
data[data['smile'].isin(new)].to_csv('../Descriptor result/new.csv')
pd.DataFrame(delet).to_csv('../Descriptor result/delet.csv')
print(len(delet))