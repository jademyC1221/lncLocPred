# -*- coding: utf-8 -*-

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import  Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
import numpy as np
from scipy.io import loadmat

#loocv for the model lncLocPredtp8, pertp8fsc.mat is obtained from Fscore.m.
#numtp8 is the optimal feature dimention of tp8.

datadict = loadmat("pertp8fsc.mat")  
data = datadict.get('pertp8fsc')    
labeldict = loadmat("label.mat")  
label = labeldict.get('label')   
X=data
Y=label
numtp8=7228
data=X[:,0:7228]

loocv = LeaveOneOut()
steps = []
steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
steps.append(('LR', LogisticRegression(C =1000, multi_class='multinomial', solver='lbfgs')))
model = Pipeline(steps)
result = cross_val_score(model, data, Y, cv=loocv)
print(result.mean())
acc=result.mean()
print("evaluate result：%.7f%% (%.7f%%)" % (result.mean() * 100, result.std() * 100))
loocv.get_n_splits(data)
a=loocv._iter_test_indices(data)
predictionI=[]
for train_index, test_index in loocv.split(data):
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    model.fit(X_train,y_train)
    predictioni=model.predict(X_test)
    predictionj=model.predict_proba(X_test)
    predictionI.append(predictioni)
    predictionJ.append(predictionj)
PredictionI=np.array(predictionI)  

#Sn，Sp，MCC，OA calculated according to definition in paper.
import math
a1=0;a2=0;b1=0;b2=0;c1=0;c2=0;d1=0;d2=0;
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
print('accuracy_score：',accuracy_score(Y,Prediction))
print('confusion_matrix：',confusion_matrix(Y,Prediction))
print('classification_report：',classification_report(Y,Prediction,digits=4))
