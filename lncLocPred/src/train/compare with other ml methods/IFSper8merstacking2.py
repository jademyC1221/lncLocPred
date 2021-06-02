# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 11:29:35 2019

@author: CMJ
"""


from scipy.io import loadmat
datadict = loadmat("lnc8mernor655CL_2.mat")
data = datadict.get('lnc8mernor655CL_2')  
labeldict = loadmat("label.mat")  
label = labeldict.get('label') 
Y=label

#每个特征集都进行5次5折交叉验证,直接填入参数
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import  Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.svm import SVC
from mlxtend.classifier import StackingClassifier
import numpy as np
##用455个样本调参
#data_x = data
#data_y = label
#print(np.hstack((data_x, data_y)))
#indices = np.random.permutation(data_x.shape[0]) 
#rand_data_x = data_x[indices]
#rand_data_y = data_y[indices]
##print(np.hstack((rand_data_x, rand_data_y)))
#traindata = rand_data_x[0:455,:]
#trainlabel = rand_data_y[0:455,:]
#scaler = MinMaxScaler(feature_range=(-1, 1))
#rescaledX = scaler.fit_transform(traindata)
#
#scaler = MinMaxScaler(feature_range=(-1, 1))
#rescaledX = scaler.fit_transform(traindata)
#from sklearn.model_selection import GridSearchCV
#scoring = 'accuracy'
#num_folds = 5
#seed = 7
#param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]}
#model = KNeighborsClassifier()
#kfold = KFold(n_splits=num_folds, random_state=seed)
#grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
#grid_result = grid.fit(X=rescaledX, y=trainlabel)
#
#print('最优：%s 使用%s' % (grid_result.best_score_, grid_result.best_params_))
#cv_results = zip(grid_result.cv_results_['mean_test_score'],
#                 grid_result.cv_results_['std_test_score'],
#                 grid_result.cv_results_['params'])
#for mean, std, param in cv_results:
#    print('%f (%f) with %r' % (mean, std, param))
##最优：0.6615384615384615 使用{'n_neighbors': 17}    

clf1 = KNeighborsClassifier(n_neighbors=17)
clf2 = RandomForestClassifier(n_estimators=100,  max_features=9)
clf3 = GaussianNB()
clf4 = xgb.XGBClassifier(learning_rate =0.1,n_estimators=50,max_depth=7,min_child_weight=4,
            gamma=0.5,subsample=0.9,colsample_bytree=0.7,objective= 'multi:softmax',num_class=4)
clf5 = SVC(C=128.0, gamma = 3.0517578125e-05, kernel='rbf',probability=True)
lr = LogisticRegression(C =100, multi_class='multinomial', solver='lbfgs')
steps = []
steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
steps.append(('Stacking', StackingClassifier(classifiers=[clf1, clf2, clf3, clf4, clf5],use_probas=True,
                          average_probas=True,meta_classifier=lr)))
model = Pipeline(steps)  

import math
percent = [x/100 for x in range(2,102,2)]
Num=[]
for i in range(len(percent)):
    Num.append(math.ceil(64855*percent[i]))

OUT1=list()
OUT2=list()
num_folds = 5
for i in range(0,20,1):
    X=data[:, 0:Num[i]]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    rescaledX = scaler.fit_transform(X)
    
    resultI=[]    
    for j in range(5):
        kf = KFold(n_splits=num_folds,shuffle=True)
        resultj = cross_val_score(model, rescaledX, Y, cv=kf)
        resultI.append(resultj.mean())
    acci = np.array(resultI)
    print('加入第%d个准确率 : %s' % (i , acci.mean()))
    OUT1.append(Num[i])
    OUT2.append(acci.mean())

OUT3=list()
OUT4=list()
num_folds = 5
for i in range(20,40,1):
    X=data[:, 0:Num[i]]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    rescaledX = scaler.fit_transform(X)
    
    resultI=[]    
    for j in range(5):
        kf = KFold(n_splits=num_folds,shuffle=True)
        resultj = cross_val_score(model, rescaledX, Y, cv=kf)
        resultI.append(resultj.mean())
    acci = np.array(resultI)
    print('加入第%d个准确率 : %s' % (i , acci.mean()))
    OUT3.append(Num[i])
    OUT4.append(acci.mean())

OUT5=list()
OUT6=list()
num_folds = 5
for i in range(40,50,1):
    X=data[:, 0:Num[i]]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    rescaledX = scaler.fit_transform(X)
    
    resultI=[]    
    for j in range(5):
        kf = KFold(n_splits=num_folds,shuffle=True)
        resultj = cross_val_score(model, rescaledX, Y, cv=kf)
        resultI.append(resultj.mean())
    acci = np.array(resultI)
    print('加入第%d个准确率 : %s' % (i , acci.mean()))
    OUT5.append(Num[i])
    OUT6.append(acci.mean())

ifs8merout1=OUT1+OUT3+OUT5
ifs8merout2=OUT2+OUT4+OUT6
#合并
IFS00 = np.vstack((ifs8merout1,ifs8merout2)).T
#IFS00 = np.vstack((IFS001,IFS002))
IFS01 =np.argsort(-IFS00[:,1])
IFSorder0 = IFS00[IFS01].tolist()
Stackingifs8perorder0 = IFSorder0

#保存数据
import pickle
output = open('StackingLRIFSpercent655.pkl','wb')
pickle.dump(data,output)
pickle.dump(label,output)
pickle.dump(ifs8merout1,output)
pickle.dump(ifs8merout2,output)
pickle.dump(IFS00,output)
pickle.dump(Stackingifs8perorder0,output)
output.close()

#读取数据
import pickle
pkl_file=open('StackingLRIFSpercent655.pkl','rb')
data=pickle.load(pkl_file)
label=pickle.load(pkl_file)
ifs8merout1=pickle.load(pkl_file)
ifs8merout2=pickle.load(pkl_file)
IFS00=pickle.load(pkl_file)
Stackingifs8perorder0=pickle.load(pkl_file)
pkl_file.close()



#保存数据
import pickle
output = open('StackingLRIFSpercent655OUT12.pkl','wb')
pickle.dump(data,output)
pickle.dump(label,output)
pickle.dump(OUT1,output)
pickle.dump(OUT2,output)
output.close()

#读取数据
import pickle
pkl_file=open('StackingLRIFSpercent655OUT12.pkl','rb')
data=pickle.load(pkl_file)
label=pickle.load(pkl_file)
OUT1=pickle.load(pkl_file)
OUT2=pickle.load(pkl_file)
pkl_file.close()

#保存数据
import pickle
output = open('StackingLRIFSpercent655OUT34.pkl','wb')
pickle.dump(OUT3,output)
pickle.dump(OUT4,output)
output.close()

#读取数据
import pickle
pkl_file=open('StackingLRIFSpercent655OUT34.pkl','rb')
OUT3=pickle.load(pkl_file)
OUT4=pickle.load(pkl_file)
pkl_file.close()

#保存数据
import pickle
output = open('StackingLRIFSpercent655OUT56.pkl','wb')
pickle.dump(OUT5,output)
pickle.dump(OUT6,output)
output.close()

#读取数据
import pickle
pkl_file=open('StackingLRIFSpercent655OUT56.pkl','rb')
OUT5=pickle.load(pkl_file)
OUT6=pickle.load(pkl_file)
pkl_file.close()



#用455个样本调参
data_x = data
data_y = label
print(np.hstack((data_x, data_y)))
indices = np.random.permutation(data_x.shape[0]) 
rand_data_x = data_x[indices]
rand_data_y = data_y[indices]
#print(np.hstack((rand_data_x, rand_data_y)))
traindata = rand_data_x[0:455,:]
trainlabel = rand_data_y[0:455,:]
scaler = MinMaxScaler(feature_range=(-1, 1))
rescaledX = scaler.fit_transform(traindata)

scaler = MinMaxScaler(feature_range=(-1, 1))
rescaledX = scaler.fit_transform(traindata)
from sklearn.model_selection import GridSearchCV
scoring = 'accuracy'
num_folds = 5
seed = 7
param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]}
model = KNeighborsClassifier()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(X=rescaledX, y=trainlabel)

print('最优：%s 使用%s' % (grid_result.best_score_, grid_result.best_params_))
cv_results = zip(grid_result.cv_results_['mean_test_score'],
                 grid_result.cv_results_['std_test_score'],
                 grid_result.cv_results_['params'])
for mean, std, param in cv_results:
    print('%f (%f) with %r' % (mean, std, param))
##最优：0.6967032967032967 使用{'n_neighbors': 3}  


from scipy.io import loadmat
datadict = loadmat("pertp8Stackingfsc.mat")  
data = datadict.get('pertp8Stackingfsc')     
Y=label
clf1 = KNeighborsClassifier(n_neighbors=3)
clf2 = RandomForestClassifier(n_estimators=100,  max_features=9)
clf3 = GaussianNB()
clf4 = xgb.XGBClassifier(learning_rate =0.1,n_estimators=50,max_depth=7,min_child_weight=4,
            gamma=0.5,subsample=0.9,colsample_bytree=0.7,objective= 'multi:softmax',num_class=4)
clf5 = SVC(C=128.0, gamma = 3.0517578125e-05, kernel='rbf',probability=True)
lr = LogisticRegression(C =1000, multi_class='multinomial', solver='lbfgs')
steps = []
steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
steps.append(('Stacking', StackingClassifier(classifiers=[clf1, clf2, clf3, clf4, clf5],use_probas=True,
                          average_probas=True,meta_classifier=lr)))
model = Pipeline(steps)   
import math
percent = [x/100 for x in range(2,102,2)]
Num=[]
for i in range(len(percent)):
    Num.append(math.ceil(13222*percent[i]))

OUT1=list()
OUT2=list()
num_folds = 5
for i in range(0,39,1):
    X=data[:, 0:Num[i]]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    rescaledX = scaler.fit_transform(X)
    
    resultI=[]    
    for j in range(5):
        kf = KFold(n_splits=num_folds,shuffle=True)
        resultj = cross_val_score(model, rescaledX, Y, cv=kf)
        resultI.append(resultj.mean())
    acci = np.array(resultI)
    print('加入第%d个准确率 : %s' % (i , acci.mean()))
    OUT1.append(Num[i])
    OUT2.append(acci.mean())

OUT3=list()
OUT4=list()
num_folds = 5
for i in range(39,45,1):
    X=data[:, 0:Num[i]]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    rescaledX = scaler.fit_transform(X)
    
    resultI=[]    
    for j in range(5):
        kf = KFold(n_splits=num_folds,shuffle=True)
        resultj = cross_val_score(model, rescaledX, Y, cv=kf)
        resultI.append(resultj.mean())
    acci = np.array(resultI)
    print('加入第%d个准确率 : %s' % (i , acci.mean()))
    OUT3.append(Num[i])
    OUT4.append(acci.mean())

OUT32=list()
OUT42=list()
num_folds = 5
for i in range(45,50,1):
    X=data[:, 0:Num[i]]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    rescaledX = scaler.fit_transform(X)
    
    resultI=[]    
    for j in range(5):
        kf = KFold(n_splits=num_folds,shuffle=True)
        resultj = cross_val_score(model, rescaledX, Y, cv=kf)
        resultI.append(resultj.mean())
    acci = np.array(resultI)
    print('加入第%d个准确率 : %s' % (i , acci.mean()))
    OUT32.append(Num[i])
    OUT42.append(acci.mean())

ifspertp8out1=OUT1+OUT3+OUT32
ifspertp8out2=OUT2+OUT4+OUT42
IFS00 = np.vstack((ifspertp8out1,ifspertp8out2)).T
IFS01 =np.argsort(-IFS00[:,1])
IFSorder0 = IFS00[IFS01].tolist()
ifspertp8Stackingorder0 = IFSorder0


#保存数据
import pickle
output = open('ifspertp8StackingOUT12.pkl','wb')
pickle.dump(data,output)
pickle.dump(label,output)
pickle.dump(OUT1,output)
pickle.dump(OUT2,output)
output.close()

#读取数据
import pickle
pkl_file=open('ifspertp8StackingOUT12.pkl','rb')
data=pickle.load(pkl_file)
label=pickle.load(pkl_file)
OUT1=pickle.load(pkl_file)
OUT2=pickle.load(pkl_file)
pkl_file.close()

#保存数据
import pickle
output = open('ifspertp8StackingOUT342.pkl','wb')
pickle.dump(OUT32,output)
pickle.dump(OUT42,output)
output.close()

#读取数据
import pickle
pkl_file=open('ifspertp8StackingOUT342.pkl','rb')
OUT32=pickle.load(pkl_file)
OUT42=pickle.load(pkl_file)
pkl_file.close()


#保存数据
import pickle
output = open('IFSpertp8Stacking.pkl','wb')
pickle.dump(data,output)
pickle.dump(label,output)
pickle.dump(ifspertp8out1,output)
pickle.dump(ifspertp8out2,output)
pickle.dump(ifspertp8Stackingorder0,output)
output.close()

#读取数据
import pickle
pkl_file=open('IFSpertp8Stacking.pkl','rb')
data=pickle.load(pkl_file)
label=pickle.load(pkl_file)
ifspertp8out1=pickle.load(pkl_file)
ifspertp8out2=pickle.load(pkl_file)
pickle.dump(ifspertp8Stackingorder0,output)
pkl_file.close()


'''留一法'''
from sklearn.model_selection import LeaveOneOut
import numpy as np
X=data
Y=label
data=X[:,0:8198]
#留一法
loocv = LeaveOneOut()
clf1 = KNeighborsClassifier(n_neighbors=3)
clf2 = RandomForestClassifier(n_estimators=100,  max_features=9)
clf3 = GaussianNB()
clf4 = xgb.XGBClassifier(learning_rate =0.1,n_estimators=50,max_depth=7,min_child_weight=4,
            gamma=0.5,subsample=0.9,colsample_bytree=0.7,objective= 'multi:softmax',num_class=4)
clf5 = SVC(C=128.0, gamma = 3.0517578125e-05, kernel='rbf',probability=True)
lr = LogisticRegression(C =1000, multi_class='multinomial', solver='lbfgs')
steps = []
steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
steps.append(('Stacking', StackingClassifier(classifiers=[clf1, clf2, clf3, clf4, clf5],use_probas=True,
                          average_probas=True,meta_classifier=lr)))
model = Pipeline(steps)

result = cross_val_score(model, data, Y, cv=loocv) 
print(result.mean())
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
print("算法评估结果：%.4f%% (%.4f%%)" % (result.mean() * 100, result.std() * 100))

#计算Sn，Sp，MCC，OA
import math
a1=0;a2=0;b1=0;b2=0;c1=0;c2=0;d1=0;d2=0;
#num1=[154,417,43,30];num2=[490,227,601,614];num1=np.array(num1);num2=np.array(num2)
for i in range(0,644,1):
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
sn1=1-(a1/154) ; sp1=1-(a2/490)
sn2=1-(b1/417) ; sp2=1-(b2/227)
sn3=1-(c1/43)  ; sp3=1-(c2/601)
sn4=1-(d1/30)  ; sp4=1-(d2/614)
mcc1=(sn1+sp1-1)/math.sqrt((sn1+(a2/154))*(sp1+(a1/490)))
mcc2=(sn2+sp2-1)/math.sqrt((sn2+(b2/417))*(sp2+(b1/227)))
mcc3=(sn3+sp3-1)/math.sqrt((sn3+(c2/43))*(sp3+(c1/601)))
mcc4=(sn4+sp4-1)/math.sqrt((sn4+(d2/30))*(sp4+(d1/614)))
    
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
output = open('IFSpertp8Stackingloocv.pkl','wb')   
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
pkl_file=open('IFSpertp8Stackingloocv.pkl','rb')   
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