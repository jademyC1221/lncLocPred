# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 19:51:40 2019

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
from sklearn.naive_bayes import GaussianNB
import numpy as np
import math
percent = [x/100 for x in range(2,102,2)]
Num=[]
for i in range(len(percent)):
    Num.append(math.ceil(64855*percent[i]))

OUT1=list()
OUT2=list()
num_folds = 5
for i in range(0,30,1):
    X=data[:, 0:Num[i]]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    rescaledX = scaler.fit_transform(X)
    
    resultI=[]    
    for j in range(5):
        kf = KFold(n_splits=num_folds,shuffle=True)
        model = GaussianNB()
        resultj = cross_val_score(model, rescaledX, Y, cv=kf)
        resultI.append(resultj.mean())
    acci = np.array(resultI)
    print('加入第%d个准确率 : %s' % (i , acci.mean()))
    OUT1.append(Num[i])
    OUT2.append(acci.mean())

OUT3=list()
OUT4=list()
num_folds = 5
for i in range(30,50,1):
    X=data[:, 0:Num[i]]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    rescaledX = scaler.fit_transform(X)
    
    resultI=[]    
    for j in range(5):
        kf = KFold(n_splits=num_folds,shuffle=True)
        model = GaussianNB()
        resultj = cross_val_score(model, rescaledX, Y, cv=kf)
        resultI.append(resultj.mean())
    acci = np.array(resultI)
    print('加入第%d个准确率 : %s' % (i , acci.mean()))
    OUT3.append(Num[i])
    OUT4.append(acci.mean()) 

ifs8merout1=OUT1+OUT3
ifs8merout2=OUT2+OUT4
#合并
IFS00 = np.vstack((ifs8merout1,ifs8merout2)).T
#IFS00 = np.vstack((IFS001,IFS002))
IFS01 =np.argsort(-IFS00[:,1])
IFSorder0 = IFS00[IFS01].tolist()
Bayesifs8perorder0 = IFSorder0

#保存数据
import pickle
output = open('BayesLRIFSpercent655.pkl','wb')
pickle.dump(data,output)
pickle.dump(label,output)
pickle.dump(ifs8merout1,output)
pickle.dump(ifs8merout2,output)
pickle.dump(IFS00,output)
pickle.dump(Bayesifs8perorder0,output)
output.close()

#读取数据
import pickle
pkl_file=open('BayesLRIFSpercent655.pkl','rb')
data=pickle.load(pkl_file)
label=pickle.load(pkl_file)
ifs8merout1=pickle.load(pkl_file)
ifs8merout2=pickle.load(pkl_file)
IFS00=pickle.load(pkl_file)
Bayesifs8perorder0=pickle.load(pkl_file)
pkl_file.close()

#保存数据
import pickle
output = open('BayesLRIFSpercent655OUT34.pkl','wb')
pickle.dump(OUT3,output)
pickle.dump(OUT4,output)
output.close()

#读取数据
import pickle
pkl_file=open('BayesLRIFSpercent655OUT34.pkl','rb')
OUT3=pickle.load(pkl_file)
OUT4=pickle.load(pkl_file)
pkl_file.close()



'''然后和triplet，psednc特征结合'''
from scipy.io import loadmat
datadict = loadmat("pertp8bayesfsc.mat")  
data = datadict.get('pertp8bayesfsc')     
labeldict = loadmat("label.mat")  
label = labeldict.get('label') 
Y=label

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
import numpy as np
import math
percent = [x/100 for x in range(2,102,2)]
Num=[]
for i in range(len(percent)):
    Num.append(math.ceil(26193*percent[i]))

OUT1=list()
OUT2=list()
num_folds = 5
for i in range(0,50,1):
    X=data[:, 0:Num[i]]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    rescaledX = scaler.fit_transform(X)
    
    resultI=[]    
    for j in range(5):
        kf = KFold(n_splits=num_folds,shuffle=True)
        model = GaussianNB()
        resultj = cross_val_score(model, rescaledX, Y, cv=kf)
        resultI.append(resultj.mean())
    acci = np.array(resultI)
    print('加入第%d个准确率 : %s' % (i , acci.mean()))
    OUT1.append(Num[i])
    OUT2.append(acci.mean())

ifspertp8out1=OUT1
ifspertp8out2=OUT2
IFS00 = np.vstack((OUT1,OUT2)).T
IFS01 =np.argsort(-IFS00[:,1])
IFSorder0 = IFS00[IFS01].tolist()
ifspertp8Bayesorder0 = IFSorder0

#保存数据
import pickle
output = open('IFSpertp8Bayes.pkl','wb')
pickle.dump(data,output)
pickle.dump(label,output)
pickle.dump(ifspertp8out1,output)
pickle.dump(ifspertp8out2,output)
pickle.dump(ifspertp8Bayesorder0,output)
output.close()

#读取数据
import pickle
pkl_file=open('IFSpertp8Bayes.pkl','rb')
data=pickle.load(pkl_file)
label=pickle.load(pkl_file)
ifspertp8out1=pickle.load(pkl_file)
ifspertp8out2=pickle.load(pkl_file)
pickle.dump(ifspertp8Bayesorder0,output)
pkl_file.close()


from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import  Pipeline
X = data
data = X[:,0:23050]
loocv = LeaveOneOut()
steps = []
steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
steps.append(('GaussianNB', GaussianNB()))
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
print("算法评估结果：%.4f%% (%.4f%%)" % (result.mean() * 100, result.std() * 100))

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
output = open('IFSpertp8Bayesloocv.pkl','wb')
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
pkl_file=open('IFSpertp8Bayesloocv.pkl','rb')
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
