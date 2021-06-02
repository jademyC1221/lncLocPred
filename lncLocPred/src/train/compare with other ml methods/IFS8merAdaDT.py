# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 20:28:03 2019

@author: chenmeijun
"""


from sklearn.feature_selection import *
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import  Pipeline
from sklearn.model_selection import LeaveOneOut
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.io import loadmat

datadict = loadmat("lnc8mernor655CL.mat")
data = datadict.get('lnc8mernor655CL')   
labeldict = loadmat("label.mat")  
label = labeldict.get('label') 

threshold = 0
def test_VarianceThreshold(X,threshold):
    selector = VarianceThreshold(threshold)
    selector.fit(X)
    print("Variances is %s" % selector.variances_)
    print("After transform is %s" % selector.transform(X))
    print("The surport is %s" % selector.get_support(True))
    print("After reverse transform is %s" %selector.inverse_transform(selector.transform(X)))
    return selector.variances_,selector.transform(X)
variances,data2 = test_VarianceThreshold(X=data,threshold=threshold)

data = data2

#用455个样本调参
data_x = data
data_y = label
print(np.hstack((data_x, data_y)))
indices = np.random.permutation(data_x.shape[0])  
rand_data_x = data_x[indices]
rand_data_y = data_y[indices]
print(np.hstack((rand_data_x, rand_data_y)))

traindata = rand_data_x[0:455,:]
trainlabel = rand_data_y[0:455,:]

#将数据规定在[-1,1]之间
scaler = MinMaxScaler(feature_range=(-1, 1))
rescaledX = scaler.fit_transform(traindata)

# 调参
num_folds = 5
seed = 7
kfold = KFold(n_splits=num_folds, random_state=seed)
from sklearn.model_selection import GridSearchCV

param_grid = {'max_features':range(3,9,1),'min_samples_leaf':range(1,10,1),
                'max_depth':range(3,9,1)}
grid = GridSearchCV(estimator = DecisionTreeClassifier(class_weight=None, criterion='gini', 
            max_leaf_nodes=None, min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=1, splitter='random'), 
                       param_grid = param_grid, scoring='accuracy',cv=5)
grid_result = grid.fit(X=rescaledX, y=trainlabel)
grid.grid_scores_
print('最优：%s 使用%s' % (grid_result.best_score_, grid_result.best_params_))
cv_results = zip(grid_result.cv_results_['mean_test_score'],
                 grid_result.cv_results_['std_test_score'],
                 grid_result.cv_results_['params'])
for mean, std, param in cv_results:
    print('%f (%f) with %r' % (mean, std, param))
#最优：0.6395604395604395 使用{'max_depth': 5, 'max_features': 7, 'min_samples_leaf': 2}
    
data_x = data
data_y = label
print(np.hstack((data_x, data_y)))
indices = np.random.permutation(data_x.shape[0])  
rand_data_x = data_x[indices]
rand_data_y = data_y[indices]
print(np.hstack((rand_data_x, rand_data_y)))

traindata = rand_data_x[0:455,:]
trainlabel = rand_data_y[0:455,:]

scaler = MinMaxScaler(feature_range=(-1, 1))
rescaledX = scaler.fit_transform(traindata)
set_printoptions(precision=3)
print(rescaledX)
# 调参
num_folds = 5
seed = 7
kfold = KFold(n_splits=num_folds, random_state=seed)
from sklearn.model_selection import GridSearchCV
scoring = 'accuracy'
param_grid = {'n_estimators': [10, 30, 50, 70, 90, 100]}
model = AdaBoostClassifier(DecisionTreeClassifier(class_weight=None, criterion='gini', 
            max_leaf_nodes=None, min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=1, splitter='random',max_depth=5,max_features=7,min_samples_leaf=2),algorithm='SAMME.R')
#LogisticRegression(),algorithm='SAMME.R'
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(X=rescaledX, y=trainlabel)

print('最优：%s 使用%s' % (grid_result.best_score_, grid_result.best_params_))
cv_results = zip(grid_result.cv_results_['mean_test_score'],
                 grid_result.cv_results_['std_test_score'],
                 grid_result.cv_results_['params'])
for mean, std, param in cv_results:
    print('%f (%f) with %r' % (mean, std, param))
#最优：0.6 使用{'n_estimators': 100}
    
#每个特征集都进行5次5折交叉验证,直接填入参数
Y=label
OUT1=list()
OUT2=list()
num_folds = 5
for i in range(500,66000,500):
    X=data[:, 0:i]
    #将数据规定在[-1,1]之间，已经from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # 数据转换
    rescaledX = scaler.fit_transform(X)
    resultI=[]    
    for j in range(5):
        kf = KFold(n_splits=num_folds,shuffle=True)
        model = AdaBoostClassifier(DecisionTreeClassifier(class_weight=None, criterion='gini', 
            max_leaf_nodes=None, min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=1, splitter='random',max_depth=5,max_features=7,min_samples_leaf=2),algorithm='SAMME.R',n_estimators=100)
        resultj = cross_val_score(model, rescaledX, Y, cv=kf)
        resultI.append(resultj.mean())
    acci = np.array(resultI)
    print('加入第%d个准确率 : %s' % (i , acci.mean()))
    OUT1.append(i)
    OUT2.append(acci.mean())

import numpy as np
OUT1=np.array(OUT1)
OUT2=np.array(OUT2)
#合并
IFS00 = np.vstack((OUT1,OUT2)).T
IFS01 =np.argsort(-IFS00[:,1])
IFSorder0 = IFS00[IFS01].tolist()
IFSorder0 = np.array(IFSorder0)
ifsadaDT8merorder0=IFSorder0


#
OUT3=list()
OUT4=list()
num_folds = 5
for i in range(2500,7100,100):
    X=data[:, 0:i]
    #将数据规定在[-1,1]之间，已经from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # 数据转换
    rescaledX = scaler.fit_transform(X)
    resultI=[]    
    for j in range(20):
        kf = KFold(n_splits=num_folds,shuffle=True)
        model = AdaBoostClassifier(DecisionTreeClassifier(class_weight=None, criterion='gini', 
            max_leaf_nodes=None, min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=1, splitter='random',max_depth=5,max_features=7,min_samples_leaf=2),algorithm='SAMME.R',n_estimators=100)
        resultj = cross_val_score(model, rescaledX, Y, cv=kf)
        resultI.append(resultj.mean())
    acci = np.array(resultI)
    print('加入第%d个准确率 : %s' % (i , acci.mean()))
    OUT3.append(i)
    OUT4.append(acci.mean())

import numpy as np
OUT3=np.array(OUT3)
OUT4=np.array(OUT4)
#合并
IFS10 = np.vstack((OUT3,OUT4)).T
IFS01 =np.argsort(-IFS10[:,1])
IFSorder0 = IFS10[IFS01].tolist()
IFSorder0 = np.array(IFSorder0)
ifsadaDT8merorder1=IFSorder0

OUT5=list()
OUT6=list()
num_folds = 5
for i in range(4300,7020,20):
    X=data[:, 0:i]
    #将数据规定在[-1,1]之间，已经from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # 数据转换
    rescaledX = scaler.fit_transform(X)
    resultI=[]    
    for j in range(20):
        kf = KFold(n_splits=num_folds,shuffle=True)
        model = AdaBoostClassifier(DecisionTreeClassifier(class_weight=None, criterion='gini', 
            max_leaf_nodes=None, min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=1, splitter='random',max_depth=5,max_features=7,min_samples_leaf=2),algorithm='SAMME.R',n_estimators=100)
        resultj = cross_val_score(model, rescaledX, Y, cv=kf)
        resultI.append(resultj.mean())
    acci = np.array(resultI)
    print('加入第%d个准确率 : %s' % (i , acci.mean()))
    OUT5.append(i)
    OUT6.append(acci.mean())

import numpy as np
OUT5=np.array(OUT5)
OUT6=np.array(OUT6)
#合并
IFS20 = np.vstack((OUT5,OUT6)).T
IFS01 =np.argsort(-IFS20[:,1])
IFSorder0 = IFS20[IFS01].tolist()
IFSorder0 = np.array(IFSorder0)
ifsadaDT8merorder2=IFSorder0

OUT7_1=list()
OUT8_1=list()
num_folds = 5
for i in range(4540,4585,5):
    X=data[:, 0:i]
    #将数据规定在[-1,1]之间，已经from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # 数据转换
    rescaledX = scaler.fit_transform(X)
    resultI=[]    
    for j in range(20):
        kf = KFold(n_splits=num_folds,shuffle=True)
        model = AdaBoostClassifier(DecisionTreeClassifier(class_weight=None, criterion='gini', 
            max_leaf_nodes=None, min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=1, splitter='random',max_depth=5,max_features=7,min_samples_leaf=2),algorithm='SAMME.R',n_estimators=100)
        resultj = cross_val_score(model, rescaledX, Y, cv=kf)
        resultI.append(resultj.mean())
    acci = np.array(resultI)
    print('加入第%d个准确率 : %s' % (i , acci.mean()))
    OUT7_1.append(i)
    OUT8_1.append(acci.mean())
OUT7_2=list()
OUT8_2=list()
num_folds = 5
for i in range(5600,5645,5):
    X=data[:, 0:i]
    #将数据规定在[-1,1]之间，已经from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # 数据转换
    rescaledX = scaler.fit_transform(X)
    resultI=[]    
    for j in range(20):
        kf = KFold(n_splits=num_folds,shuffle=True)
        model = AdaBoostClassifier(DecisionTreeClassifier(class_weight=None, criterion='gini', 
            max_leaf_nodes=None, min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=1, splitter='random',max_depth=5,max_features=7,min_samples_leaf=2),algorithm='SAMME.R',n_estimators=100)
        resultj = cross_val_score(model, rescaledX, Y, cv=kf)
        resultI.append(resultj.mean())
    acci = np.array(resultI)
    print('加入第%d个准确率 : %s' % (i , acci.mean()))
    OUT7_2.append(i)
    OUT8_2.append(acci.mean())

import numpy as np
OUT7_1=np.array(OUT7_1)
OUT8_1=np.array(OUT8_1)
OUT7_2=np.array(OUT7_2)
OUT8_2=np.array(OUT8_2)
#合并
IFS00 = np.vstack((OUT7_1,OUT8_1)).T
IFS01 = np.vstack((OUT7_2,OUT8_2)).T
IFS30 = np.vstack((IFS00,IFS01))
IFS01 =np.argsort(-IFS30[:,1])
IFSorder0 = IFS30[IFS01].tolist()
IFSorder0 = np.array(IFSorder0)
ifsadaDT8merorder3=IFSorder0

#保存数据
import pickle
output = open('IFS8merAdaDT655.pkl','wb')
pickle.dump(data,output)
pickle.dump(label,output)
pickle.dump(IFS00,output)
pickle.dump(IFS10,output)
pickle.dump(IFS20,output)
pickle.dump(IFS30,output)
pickle.dump(ifsadaDT8merorder0,output)
pickle.dump(ifsadaDT8merorder1,output)
pickle.dump(ifsadaDT8merorder2,output)
pickle.dump(ifsadaDT8merorder3,output)
output.close()

#读取数据
import pickle
pkl_file=open('IFS8merAdaDT655.pkl','rb')
data=pickle.load(pkl_file)
label=pickle.load(pkl_file)
IFS00=pickle.load(pkl_file)
IFS10=pickle.load(pkl_file)
IFS20=pickle.load(pkl_file)
IFS30=pickle.load(pkl_file)
ifsadaDT8merorder0=pickle.load(pkl_file)
ifsadaDT8merorder1=pickle.load(pkl_file)
ifsadaDT8merorder2=pickle.load(pkl_file)
ifsadaDT8merorder3=pickle.load(pkl_file)
pkl_file.close()
