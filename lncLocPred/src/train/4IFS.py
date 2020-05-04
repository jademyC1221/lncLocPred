# -*- coding: utf-8 -*-
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
import math
from sklearn.model_selection import GridSearchCV
from scipy.io import loadmat
#load label
labeldict = loadmat("label.mat")  
label = labeldict.get('label') 
Y = label

'''load kmer feature data after sorted by binomial distribution'''
'''1.k=5'''
datadict = loadmat("lnc5mernor655CL.mat")  
data = datadict.get('lnc5mernor655CL')     

percent = [x/100 for x in range(1,101,1)]
Num=[]
for i in range(len(percent)):
    Num.append(math.ceil(1024*percent[i]))
ifs5merout1=list()
ifs5merout2=list()
ifs5merout3=list()
num_folds = 5
for i in range(len(percent)):
    X=data[:, 0:Num[i]]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    rescaledX = scaler.fit_transform(X)
    
    resultIgs=[]
    resultIparams=[]
    for x in range(5):
        num_folds = 5
        kf = KFold(n_splits=num_folds,shuffle=True)
        param_grid = {}
        #正则化就是为了解决过拟合问题(overfitting problem)。c越小，正则化越强,C越大，正则化越弱，泛化能力越弱。
        param_grid['C'] = [0.001, 0.01, 0.1, 1.0, 10, 100, 1000]
        param_grid['solver'] = ['newton-cg','lbfgs']
        param_grid['multi_class'] = ['ovr','multinomial']
        model = LogisticRegression()
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=kf)
        grid_result = grid.fit(rescaledX,label)
        print('最优 : %s 使用 %s' % (grid_result.best_score_, grid_result.best_params_))
        resultIgs.append(grid_result.best_score_)
        resultIparams.append(grid_result.best_params_)
    IFS00 = list(zip(resultIgs,resultIparams))
    IFS01 = np.argsort(-np.array(resultIgs))
    params = resultIparams[IFS01[0]]
    
    resultI=[]    
    for j in range(5):
        kf = KFold(n_splits=num_folds,shuffle=True)
        model = LogisticRegression(**params)
        resultj = cross_val_score(model, rescaledX, Y, cv=kf)
        resultI.append(resultj.mean())
    acci = np.array(resultI)
    print('加入第%d个准确率 : %s' % (i , acci.mean()))
    ifs5merout1.append(Num[i])
    ifs5merout2.append(params)
    ifs5merout3.append(acci.mean())

IFS00 = np.vstack((ifs5merout1,ifs5merout2,ifs5merout3)).T
IFS01 =np.argsort(-IFS00[:,2])
IFSorder0 = IFS00[IFS01].tolist()
ifs5perorder0 = IFSorder0



'''2.k=6'''
datadict = loadmat("lnc6mernor655CL.mat")
data = datadict.get('lnc6mernor655CL')

percent = [x/100 for x in range(1,101,1)]
Num=[]
for i in range(len(percent)):
    Num.append(math.ceil(4096*percent[i]))
ifs6merout1=list()
ifs6merout2=list()
ifs6merout3=list()
num_folds = 5
for i in range(len(percent)):
    X=data[:, 0:Num[i]]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    rescaledX = scaler.fit_transform(X)
    
    resultIgs=[]
    resultIparams=[]
    for x in range(5):
        num_folds = 5
        kf = KFold(n_splits=num_folds,shuffle=True)
        param_grid = {}
        #正则化就是为了解决过拟合问题(overfitting problem)。c越小，正则化越强,C越大，正则化越弱，泛化能力越弱。
        param_grid['C'] = [0.001, 0.01, 0.1, 1.0, 10, 100, 1000]
        param_grid['solver'] = ['newton-cg','lbfgs']
        param_grid['multi_class'] = ['ovr','multinomial']
        model = LogisticRegression()
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=kf)
        grid_result = grid.fit(rescaledX,label)
        print('最优 : %s 使用 %s' % (grid_result.best_score_, grid_result.best_params_))
        resultIgs.append(grid_result.best_score_)
        resultIparams.append(grid_result.best_params_)
    IFS00 = list(zip(resultIgs,resultIparams))
    IFS01 = np.argsort(-np.array(resultIgs))
    params = resultIparams[IFS01[0]]
    
    resultI=[]    
    for j in range(5):
        kf = KFold(n_splits=num_folds,shuffle=True)
        model = LogisticRegression(**params)
        resultj = cross_val_score(model, rescaledX, Y, cv=kf)
        resultI.append(resultj.mean())
    acci = np.array(resultI)
    print('加入第%d个准确率 : %s' % (i , acci.mean()))
    ifs6merout1.append(Num[i])
    ifs6merout2.append(params)
    ifs6merout3.append(acci.mean())

IFS00 = np.vstack((ifs6merout1,ifs6merout2,ifs6merout3)).T
IFS01 =np.argsort(-IFS00[:,2])
IFSorder0 = IFS00[IFS01].tolist()
ifs6perorder0 = IFSorder0


'''3.k=8'''
datadict = loadmat("lnc8mernor655CL_2.mat")
data = datadict.get('lnc8mernor655CL_2')  

percent = [x/100 for x in range(2,102,2)]
Num=[]
for i in range(len(percent)):
    Num.append(math.ceil(64855*percent[i]))
 
ifs8merout1=list()
ifs8merout2=list()
ifs8merout3=list()
num_folds = 5
for i in range(0,15,1):
    X=data[:, 0:Num[i]]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    rescaledX = scaler.fit_transform(X)
    
    resultIgs=[]
    resultIparams=[]
    for x in range(5):
        num_folds = 5
        kf = KFold(n_splits=num_folds,shuffle=True)
        param_grid = {}
        param_grid['C'] = [0.001, 0.01, 0.1, 1.0, 10, 100, 1000]
        param_grid['solver'] = ['newton-cg','lbfgs']
        param_grid['multi_class'] = ['ovr','multinomial']
        model = LogisticRegression()
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=kf)
        grid_result = grid.fit(rescaledX,label)
        print('最优 : %s 使用 %s' % (grid_result.best_score_, grid_result.best_params_))
        resultIgs.append(grid_result.best_score_)
        resultIparams.append(grid_result.best_params_)
    IFS00 = list(zip(resultIgs,resultIparams))
    IFS01 = np.argsort(-np.array(resultIgs))
    params = resultIparams[IFS01[0]]
    
    resultI=[]    
    for j in range(5):
        kf = KFold(n_splits=num_folds,shuffle=True)
        model = LogisticRegression(**params)
        resultj = cross_val_score(model, rescaledX, Y, cv=kf)
        resultI.append(resultj.mean())
    acci = np.array(resultI)
    print('加入第%d个准确率 : %s' % (i , acci.mean()))
    ifs8merout1.append(Num[i])
    ifs8merout2.append(params)
    ifs8merout3.append(acci.mean())

IFS00 = np.vstack((ifs8merout1,ifs8merout2,ifs8merout3)).T
IFS01 =np.argsort(-IFS00[:,2])
IFSorder0 = IFS00[IFS01].tolist()
ifs8perorder0 = IFSorder0