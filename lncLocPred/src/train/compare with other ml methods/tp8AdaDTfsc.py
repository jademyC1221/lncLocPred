# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 14:43:44 2019

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

datadict = loadmat("AdaDTtp8fsc.mat")
data = datadict.get('AdaDTtp8fsc')   
labeldict = loadmat("label.mat")  
label = labeldict.get('label') 


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
#最优：0.6879120879120879 使用{'max_depth': 4, 'max_features': 7, 'min_samples_leaf': 1}
    
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
            presort=False, random_state=1, splitter='random',max_depth=4,max_features=7,min_samples_leaf=1),algorithm='SAMME.R')
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
#最优：0.6659340659340659 使用{'n_estimators': 10}
    
#每个特征集都进行5次5折交叉验证,直接填入参数
Y=label
OUT1_1=list()
OUT2_1=list()
num_folds = 5
for i in range(200,4200,200):
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
            presort=False, random_state=1, splitter='random',max_depth=4,max_features=7,min_samples_leaf=1),algorithm='SAMME.R',n_estimators=10)
        resultj = cross_val_score(model, rescaledX, Y, cv=kf)
        resultI.append(resultj.mean())
    acci = np.array(resultI)
    print('加入第%d个准确率 : %s' % (i , acci.mean()))
    OUT1_1.append(i)
    OUT2_1.append(acci.mean())
OUT1_2=list()
OUT2_2=list()
num_folds = 5
for i in range(4200,7200,200):
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
            presort=False, random_state=1, splitter='random',max_depth=4,max_features=7,min_samples_leaf=1),algorithm='SAMME.R',n_estimators=10)
        resultj = cross_val_score(model, rescaledX, Y, cv=kf)
        resultI.append(resultj.mean())
    acci = np.array(resultI)
    print('加入第%d个准确率 : %s' % (i , acci.mean()))
    OUT1_2.append(i)
    OUT2_2.append(acci.mean())
import numpy as np
OUT1_1=np.array(OUT1_1)
OUT2_1=np.array(OUT2_1)
OUT1_2=np.array(OUT1_2)
OUT2_2=np.array(OUT2_2)

#合并
IFS00 = np.vstack((OUT1_1,OUT2_1)).T
IFS01 = np.vstack((OUT1_2,OUT2_2)).T
IFS10 = np.vstack((IFS00,IFS01))
IFS07 =np.argsort(-IFS10[:,1])
IFSorder0 = IFS10[IFS07].tolist()
IFSorder0 = np.array(IFSorder0)
ifsadaDT8merorder0=IFSorder0

OUT3=list()
OUT4=list()
num_folds = 5
for i in range(200,1800,5):
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
            presort=False, random_state=1, splitter='random',max_depth=4,max_features=7,min_samples_leaf=1),algorithm='SAMME.R',n_estimators=10)
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
IFS00 = np.vstack((OUT3,OUT4)).T
IFS07 =np.argsort(-IFS00[:,1])
IFSorder0 = IFS00[IFS07].tolist()
IFSorder0 = np.array(IFSorder0)
ifsadaDT8merorder1=IFSorder0


#保存数据
import pickle
output = open('tp8merAdaDTfsc.pkl','wb')
pickle.dump(data,output)
pickle.dump(label,output)
pickle.dump(IFS10,output)
pickle.dump(IFS00,output)
pickle.dump(ifsadaDT8merorder0,output)
pickle.dump(ifsadaDT8merorder1,output)
output.close()

import pickle
pkl_file=open('tp8merAdaDTfsc.pkl','rb')
data=pickle.load(pkl_file)
label=pickle.load(pkl_file)
IFS10=pickle.load(pkl_file)
IFS00=pickle.load(pkl_file)
ifsadaDT8merorder0=pickle.load(pkl_file)
ifsadaDT8merorder1=pickle.load(pkl_file)
pkl_file.close()