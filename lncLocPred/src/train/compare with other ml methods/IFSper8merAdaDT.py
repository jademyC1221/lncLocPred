# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 15:48:15 2020

@author: CMJ
"""


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
datadict = loadmat("lnc8mernor655CL_2.mat")
data = datadict.get('lnc8mernor655CL_2')   
labeldict = loadmat("label.mat")  
label = labeldict.get('label') 
Y=label
import math
percent = [x/100 for x in range(2,102,2)]
Num=[]
for i in range(len(percent)):
    Num.append(math.ceil(64855*percent[i]))

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
#最优：0.5978021978021978 使用{'n_estimators': 100}



OUT1=list()
OUT2=list()
OUT3=list()
num_folds = 5
for i in range(32,42,1):
    X=data[:, 0:Num[i]]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    rescaledX = scaler.fit_transform(X)
    
    resultIgs=[]
    resultIparams=[]
    for x in range(2):
        num_folds = 5
        kf = KFold(n_splits=num_folds,shuffle=True)
        param_grid = {}
        param_grid['max_features'] = list(range(3,10,1))
        param_grid['min_samples_leaf'] = list(range(1,10,1))
        param_grid['max_depth'] = list(range(3,9,1))
        model = DecisionTreeClassifier(class_weight=None, criterion='gini', 
            max_leaf_nodes=None, min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=1, splitter='random')
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
        model = AdaBoostClassifier(DecisionTreeClassifier(class_weight=None, criterion='gini', 
            max_leaf_nodes=None, min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=1, splitter='random',**params),
                                   algorithm='SAMME.R',n_estimators=100)
        resultj = cross_val_score(model, rescaledX, Y, cv=kf)
        resultI.append(resultj.mean())
    acci = np.array(resultI)
    print('加入第%d个准确率 : %s' % (i , acci.mean()))
    OUT1.append(Num[i])
    OUT2.append(params)
    OUT3.append(acci.mean())


OUT4=list()
OUT5=list()
OUT6=list()
num_folds = 5
for i in range(42,50,1):
    X=data[:, 0:Num[i]]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    rescaledX = scaler.fit_transform(X)
    
    resultIgs=[]
    resultIparams=[]
    for x in range(2):
        num_folds = 5
        kf = KFold(n_splits=num_folds,shuffle=True)
        param_grid = {}
        param_grid['max_features'] = list(range(3,10,1))
        param_grid['min_samples_leaf'] = list(range(1,10,1))
        param_grid['max_depth'] = list(range(3,9,1))
        model = DecisionTreeClassifier(class_weight=None, criterion='gini', 
            max_leaf_nodes=None, min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=1, splitter='random')
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
        model = AdaBoostClassifier(DecisionTreeClassifier(class_weight=None, criterion='gini', 
            max_leaf_nodes=None, min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=1, splitter='random',**params),
                                   algorithm='SAMME.R',n_estimators=100)
        resultj = cross_val_score(model, rescaledX, Y, cv=kf)
        resultI.append(resultj.mean())
    acci = np.array(resultI)
    print('加入第%d个准确率 : %s' % (i , acci.mean()))
    OUT4.append(Num[i])
    OUT5.append(params)
    OUT6.append(acci.mean())

ifs8merout1=OUT1+OUT4
ifs8merout2=OUT2+OUT5
ifs8merout3=OUT3+OUT6
#合并
IFS00 = np.vstack((ifs8merout1,ifs8merout2,ifs8merout3)).T
IFS01 =np.argsort(-IFS00[:,2])
IFSorder0 = IFS00[IFS01].tolist()
AdaDTifs8perorder0 = IFSorder0


#保存数据
import pickle
output = open('AdaDTIFSpercent655.pkl','wb')
pickle.dump(data,output)
pickle.dump(label,output)
pickle.dump(ifs8merout1,output)
pickle.dump(ifs8merout2,output)
pickle.dump(ifs8merout3,output)
pickle.dump(IFS00,output)
pickle.dump(AdaDTifs8perorder0,output)
output.close()

#读取数据
import pickle
pkl_file=open('AdaDTIFSpercent655.pkl','rb')
data=pickle.load(pkl_file)
label=pickle.load(pkl_file)
ifs8merout1=pickle.load(pkl_file)
ifs8merout2=pickle.load(pkl_file)
ifs8merout3=pickle.load(pkl_file)
IFS00=pickle.load(pkl_file)
AdaDTifs8perorder0=pickle.load(pkl_file)
pkl_file.close()


#保存数据
import pickle
output = open('AdaDTIFSpercent655OUT12.pkl','wb')
pickle.dump(data,output)
pickle.dump(label,output)
pickle.dump(OUT1,output)
pickle.dump(OUT2,output)
pickle.dump(OUT3,output)
output.close()

#读取数据
import pickle
pkl_file=open('AdaDTIFSpercent655OUT12.pkl','rb')
data=pickle.load(pkl_file)
label=pickle.load(pkl_file)
OUT1=pickle.load(pkl_file)
OUT2=pickle.load(pkl_file)
OUT3=pickle.load(pkl_file)
pkl_file.close()

#保存数据
import pickle
output = open('AdaDTIFSpercent655OUT34.pkl','wb')
pickle.dump(OUT4,output)
pickle.dump(OUT5,output)
pickle.dump(OUT6,output)
output.close()

#读取数据
pkl_file=open('AdaDTIFSpercent655OUT34.pkl','rb')
OUT4=pickle.load(pkl_file)
OUT5=pickle.load(pkl_file)
OUT6=pickle.load(pkl_file)
pkl_file.close()



from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import numpy as np
from scipy.io import loadmat
datadict = loadmat("pertp8AdaDTfsc.mat")  
data = datadict.get('pertp8AdaDTfsc')  
labeldict = loadmat("label.mat")  
label = labeldict.get('label') 
Y=label

import math
percent = [x/100 for x in range(1,101,1)]
Num=[]
for i in range(len(percent)):
    Num.append(math.ceil(2846*percent[i]))

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
#最优：0.6659340659340659 使用{'n_estimators': 100}

OUT1=list()
OUT2=list()
OUT3=list()
num_folds = 5
for i in range(0,50,1):
    X=data[:, 0:Num[i]]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    rescaledX = scaler.fit_transform(X)
    
    resultIgs=[]
    resultIparams=[]
    for x in range(2):
        num_folds = 5
        kf = KFold(n_splits=num_folds,shuffle=True)
        param_grid = {}
        param_grid['max_features'] = list(range(3,10,1))
        param_grid['min_samples_leaf'] = list(range(1,10,1))
        param_grid['max_depth'] = list(range(3,9,1))
        model = DecisionTreeClassifier(class_weight=None, criterion='gini', 
            max_leaf_nodes=None, min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=1, splitter='random')
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
        model = AdaBoostClassifier(DecisionTreeClassifier(class_weight=None, criterion='gini', 
            max_leaf_nodes=None, min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=1, splitter='random',**params),
                                   algorithm='SAMME.R',n_estimators=100)
        resultj = cross_val_score(model, rescaledX, Y, cv=kf)
        resultI.append(resultj.mean())
    acci = np.array(resultI)
    print('加入第%d个准确率 : %s' % (i , acci.mean()))
    OUT1.append(Num[i])
    OUT2.append(params)
    OUT3.append(acci.mean())


ifspertp8out1=OUT1
ifspertp8out2=OUT2
ifspertp8out3=OUT3

#合并
IFS00 = np.vstack((OUT1,OUT2,OUT3)).T
IFS01 =np.argsort(-IFS00[:,2])
IFSorder0 = IFS00[IFS01].tolist()
ifspertp8AdaDTorder0 = IFSorder0

#保存数据
import pickle
output = open('IFSpertp8AdaDT.pkl','wb')
pickle.dump(data,output)
pickle.dump(label,output)
pickle.dump(ifspertp8out1,output)
pickle.dump(ifspertp8out2,output)
pickle.dump(ifspertp8out3,output)
pickle.dump(ifspertp8AdaDTorder0,output)
output.close()

#读取数据
import pickle
pkl_file=open('IFSpertp8AdaDT.pkl','rb')
data=pickle.load(pkl_file)
label=pickle.load(pkl_file)
ifspertp8out1=pickle.load(pkl_file)
ifspertp8out2=pickle.load(pkl_file)
ifspertp8out3=pickle.load(pkl_file)
ifspertp8AdaDTorder0=pickle.load(pkl_file)
pkl_file.close()


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler


Y=label
data=X[:,0:684]

loocv = LeaveOneOut()
steps = []
steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
steps.append(('AdaBoost', AdaBoostClassifier(DecisionTreeClassifier(class_weight=None, criterion='gini', 
            max_leaf_nodes=None, min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=1, splitter='random',
            max_features= 7, min_samples_leaf= 1, max_depth= 8), algorithm='SAMME.R',n_estimators=100)))
model = Pipeline(steps)
result = cross_val_score(model, data, Y, cv=loocv)
acc=result.mean()

prediction=[]
for train_index, test_index in loocv.split(data):
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    model.fit(X_train,y_train)
    predictioni=model.predict(X_test)
    prediction.append(predictioni)
Prediction=np.array(prediction)
print(result.mean())
print("算法评估结果：%.3f%% (%.3f%%)" % (result.mean() * 100, result.std() * 100))
import numpy as np
#计算Sn，Sp，MCC，OA
import math
a1=0;a2=0;b1=0;b2=0;c1=0;c2=0;d1=0;d2=0;
#num1=[154,417,43,30];num2=[490,227,601,614];num1=np.array(num1);num2=np.array(num2)
for i in range(0,655,1):
    if Y[i]==1 and Prediction[i]!=1:
        a1+=1
    if Y[i]!=1 and Prediction[i]==1:
        a2+=1 
    if Y[i]==2 and Prediction[i]!=2:
        b1+=1
    if Y[i]!=2 and Prediction[i]==2:
        b2+=1 
    if Y[i]==3 and Prediction[i]!=3:
        c1+=1
    if Y[i]!=3 and Prediction[i]==3:
        c2+=1 
    if Y[i]==4 and Prediction[i]!=4:
        d1+=1
    if Y[i]!=4 and Prediction[i]==4:
        d2+=1 
sn1=1-(a1/156) ; sp1=1-(a2/499)
sn2=1-(b1/426) ; sp2=1-(b2/229)
sn3=1-(c1/43)  ; sp3=1-(c2/612)
sn4=1-(d1/30)  ; sp4=1-(d2/625)
mcc1=(sn1+sp1-1)/math.sqrt((sn1+(a2/156))*(sp1+(a1/499)))
mcc2=(sn2+sp2-1)/math.sqrt((sn2+(b2/426))*(sp2+(b1/229)))
mcc3=(sn3+sp3-1)/math.sqrt((sn3+(c2/43))*(sp3+(c1/612)))
mcc4=(sn4+sp4-1)/math.sqrt((sn4+(d2/30))*(sp4+(d1/625)))
    
SN = np.array([sn1,sn2,sn3,sn4])
SP = np.array([sp1,sp2,sp3,sp4])
MCC = np.array([mcc1,mcc2,mcc3,mcc4])


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
#多类分类效果评估
print('准确率：',accuracy_score(Y,Prediction))
print('混淆矩阵：',confusion_matrix(Y,Prediction))
print('分类报告：',classification_report(Y,Prediction,digits=4))

#保存数据
import pickle
output = open('IFSpertp8AdaDTloocv.pkl','wb')
pickle.dump(X,output)
pickle.dump(Y,output)
pickle.dump(data,output)
pickle.dump(result,output)
pickle.dump(Prediction,output)
pickle.dump(acc,output)
pickle.dump(SN,output)
pickle.dump(SP,output)
pickle.dump(MCC,output)
output.close()

#读取数据
import pickle
pkl_file=open('IFSpertp8AdaDTloocv.pkl','rb')
X=pickle.load(pkl_file)
Y=pickle.load(pkl_file)
data=pickle.load(pkl_file)
result=pickle.load(pkl_file)
Prediction=pickle.load(pkl_file)
acc=pickle.load(pkl_file)
SN=pickle.load(pkl_file)
SP=pickle.load(pkl_file)
MCC=pickle.load(pkl_file)
pkl_file.close()