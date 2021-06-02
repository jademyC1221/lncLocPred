# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 16:54:48 2019

@author: chenmeijun
"""


from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import *
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import  Pipeline
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.io import loadmat

datadict = loadmat("AdaDTtp8fsc.mat")
data = datadict.get('AdaDTtp8fsc')   
labeldict = loadmat("label.mat")  
label = labeldict.get('label') 

Y=label
data=data[:,0:215]

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
#最优：0.6527472527472528 使用{'n_estimators': 10}

'''
Adaboost调参
'''

num_folds = 5
seed = 7
kfold = KFold(n_splits=num_folds, random_state=seed)
from sklearn.model_selection import GridSearchCV
scoring = 'accuracy'
param_grid = {'n_estimators': [10, 30, 50, 70, 90, 100]}
model = AdaBoostClassifier(DecisionTreeClassifier(class_weight=None, criterion='gini', 
            max_leaf_nodes=None, min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=1, splitter='random',max_depth=3,max_features=4,min_samples_leaf=3),algorithm='SAMME.R')
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
#最优：0.6505494505494506 使用{'n_estimators': 10}

#留一法
from sklearn.preprocessing import MinMaxScaler

loocv = LeaveOneOut()
steps = []
steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
steps.append(('AdaBoost', AdaBoostClassifier(DecisionTreeClassifier(class_weight=None, criterion='gini', 
            max_leaf_nodes=None, min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=1, splitter='random',max_depth=3,max_features=4,min_samples_leaf=3),algorithm='SAMME.R',n_estimators=10)))
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
output = open('tp8AdaDTfscloocv.pkl','wb')  #9146
pickle.dump(data,output)
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
pkl_file=open('tp8AdaDTfscloocv.pkl','rb')    #9146
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

