import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import numpy as np
import pickle
def sim_tonimoto(user1, user2):
    common = []

    for item in range(len(user1)):
        if user1[item] != user2[item]:
            common.append(item)

    if len(common) == 0:
        return 0

    common_num = len(common)
    user1_num = len(user1)
    user2_num = len(user2)

    res = float(common_num) / (user1_num + user2_num - common_num)
    return res

def eucliDist(A, B):
 return np.sqrt(sum(np.power((A - B), 2)))
def ces(A, B):
    return np.power((A - B), 2)
def minmax(min,max):
    try:
        if max-min!=0:
            return lambda a : (a -min)/(max-min)
        else:
            return lambda a :a-a
    except:
        return lambda a: a - a
def Eulodice():
    data=pd.read_csv('../Descriptor result/descriptor_train.csv',index_col=0)
    min_max_scaler = preprocessing.MinMaxScaler()
    max_=data.max()
    min_=data.min()
    x_minmax = min_max_scaler.fit_transform(data)
    #print(x_minmax)
    center = x_minmax.mean(axis=0, dtype='float32')
    center_csv=pd.DataFrame(center).T
    center_csv.columns=data.columns
    result=pd.DataFrame([max_,min_])
    result=pd.concat([result,center_csv])
    result.index=('max','min','center')
    print(result)
    result.to_csv('../appdomin/appdomin.csv')
    distance = []
    for j in range(len(data)):
        point = x_minmax[j, :]
        euclidist = eucliDist(point, center)
        distance.append(euclidist)

    Thresh = []
    Thresh.append(max(distance))
    pd.DataFrame(Thresh).to_csv('../appdomin/appdomin.csv',mode='a')
Eulodice()
def appcalition():
    data2=pd.read_csv('../Descriptor result/descriptor_test.csv',index_col=0)
    data2.replace(['True', 'False'], [0, 1], inplace=True)
    data2.astype('float')
    app=pd.read_csv('../appdomin/appdomin.csv',index_col=0)
    for i in range(len(app.columns)):
        data2[app.columns[i]]=data2[app.columns[i]].apply(minmax(app.iat[1, i],app.iat[0, i]))
    print(data2)
    distance = []
    for j in range(len(data2)):
        point = data2.iloc[j, :]
        print(point.dtypes)
        euclidist = ces(np.array(app.iloc[2,:],dtype='float64'),point)
        euclidist.fillna(value=0, inplace=True)
        result=np.sqrt(sum(euclidist))
        distance.append(result)
    print(distance)
    print(data2.index)
    pd.DataFrame(distance).to_csv('../appdomin/test_distance.csv')
appcalition()
