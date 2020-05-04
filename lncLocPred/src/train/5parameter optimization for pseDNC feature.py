# -*- coding: utf-8 -*-

#PD40_655.pkl stores the PseDNC feature of benchmark which is downloaded from http://bioinformatics.hitsz.edu.cn/Pse-in-One2.0/

import pickle
pkl_file=open('PD40_655.pkl','rb')
PDten101 =pickle.load(pkl_file)
PDten103 =pickle.load(pkl_file)
PDten105 =pickle.load(pkl_file)
PDten107 =pickle.load(pkl_file)
PDten109 =pickle.load(pkl_file)
PDten301 =pickle.load(pkl_file)
PDten303 =pickle.load(pkl_file)
PDten305 =pickle.load(pkl_file)
PDten307 =pickle.load(pkl_file)
PDten309 =pickle.load(pkl_file)
PDten501 =pickle.load(pkl_file)
PDten503 =pickle.load(pkl_file)
PDten505 =pickle.load(pkl_file)
PDten507 =pickle.load(pkl_file)
PDten509 =pickle.load(pkl_file)
PDten701 =pickle.load(pkl_file)
PDten703 =pickle.load(pkl_file)
PDten705 =pickle.load(pkl_file)
PDten707 =pickle.load(pkl_file)
PDten709 =pickle.load(pkl_file)
PDten901 =pickle.load(pkl_file)
PDten903 =pickle.load(pkl_file)
PDten905 =pickle.load(pkl_file)
PDten907 =pickle.load(pkl_file)
PDten909 =pickle.load(pkl_file)
PDten1101 =pickle.load(pkl_file)
PDten1103 =pickle.load(pkl_file)
PDten1105 =pickle.load(pkl_file)
PDten1107 =pickle.load(pkl_file)
PDten1109 =pickle.load(pkl_file)
PDten1301 =pickle.load(pkl_file)
PDten1303 =pickle.load(pkl_file)
PDten1305 =pickle.load(pkl_file)
PDten1307 =pickle.load(pkl_file)
PDten1309 =pickle.load(pkl_file)
PDten1501 =pickle.load(pkl_file)
PDten1503 =pickle.load(pkl_file)
PDten1505 =pickle.load(pkl_file)
PDten1507 =pickle.load(pkl_file)
PDten1509 =pickle.load(pkl_file)
pkl_file.close()

from scipy.io import loadmat
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import  Pipeline
from sklearn.linear_model import LogisticRegression

labeldict = loadmat("label.mat")
label = labeldict.get('label')    
Y=label
ACC=list()
num_folds = 5
data=PDten101
resultI=[]    
for j in range(100):
    kf = KFold(n_splits=num_folds,shuffle=True)
    steps = []
    steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
    steps.append(('LR', LogisticRegression(multi_class='multinomial', solver='lbfgs')))
    model = Pipeline(steps)
    resultj = cross_val_score(model, data, Y, cv=kf)
    resultI.append(resultj.mean())
result = np.array(resultI)
acc = result.mean()
print('第%d次cv5准确率 : %s' % (j , result.mean()))
ACC.append(acc)
data=PDten103
resultI=[]    
for j in range(100):
    kf = KFold(n_splits=num_folds,shuffle=True)
    steps = []
    steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
    steps.append(('LR', LogisticRegression(multi_class='multinomial', solver='lbfgs')))
    model = Pipeline(steps)
    resultj = cross_val_score(model, data, Y, cv=kf)
    resultI.append(resultj.mean())
result = np.array(resultI)
acc = result.mean()
print('第%d次cv5准确率 : %s' % (j , result.mean()))
ACC.append(acc)
data=PDten105
resultI=[]    
for j in range(100):
    kf = KFold(n_splits=num_folds,shuffle=True)
    steps = []
    steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
    steps.append(('LR', LogisticRegression(multi_class='multinomial', solver='lbfgs')))
    model = Pipeline(steps)
    resultj = cross_val_score(model, data, Y, cv=kf)
    resultI.append(resultj.mean())
result = np.array(resultI)
acc = result.mean()
print('第%d次cv5准确率 : %s' % (j , result.mean()))
ACC.append(acc)
data=PDten107
resultI=[]    
for j in range(100):
    kf = KFold(n_splits=num_folds,shuffle=True)
    steps = []
    steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
    steps.append(('LR', LogisticRegression(multi_class='multinomial', solver='lbfgs')))
    model = Pipeline(steps)
    resultj = cross_val_score(model, data, Y, cv=kf)
    resultI.append(resultj.mean())
result = np.array(resultI)
acc = result.mean()
print('第%d次cv5准确率 : %s' % (j , result.mean()))
ACC.append(acc)
data=PDten109
resultI=[]    
for j in range(100):
    kf = KFold(n_splits=num_folds,shuffle=True)
    steps = []
    steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
    steps.append(('LR', LogisticRegression(multi_class='multinomial', solver='lbfgs')))
    model = Pipeline(steps)
    resultj = cross_val_score(model, data, Y, cv=kf)
    resultI.append(resultj.mean())
result = np.array(resultI)
acc = result.mean()
print('第%d次cv5准确率 : %s' % (j , result.mean()))
ACC.append(acc)

data=PDten301
resultI=[]    
for j in range(100):
    kf = KFold(n_splits=num_folds,shuffle=True)
    steps = []
    steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
    steps.append(('LR', LogisticRegression(multi_class='multinomial', solver='lbfgs')))
    model = Pipeline(steps)
    resultj = cross_val_score(model, data, Y, cv=kf)
    resultI.append(resultj.mean())
result = np.array(resultI)
acc = result.mean()
print('第%d次cv5准确率 : %s' % (j , result.mean()))
ACC.append(acc)
data=PDten303
resultI=[]    
for j in range(100):
    kf = KFold(n_splits=num_folds,shuffle=True)
    steps = []
    steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
    steps.append(('LR', LogisticRegression(multi_class='multinomial', solver='lbfgs')))
    model = Pipeline(steps)
    resultj = cross_val_score(model, data, Y, cv=kf)
    resultI.append(resultj.mean())
result = np.array(resultI)
acc = result.mean()
print('第%d次cv5准确率 : %s' % (j , result.mean()))
ACC.append(acc)
data=PDten305
resultI=[]    
for j in range(100):
    kf = KFold(n_splits=num_folds,shuffle=True)
    steps = []
    steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
    steps.append(('LR', LogisticRegression(multi_class='multinomial', solver='lbfgs')))
    model = Pipeline(steps)
    resultj = cross_val_score(model, data, Y, cv=kf)
    resultI.append(resultj.mean())
result = np.array(resultI)
acc = result.mean()
print('第%d次cv5准确率 : %s' % (j , result.mean()))
ACC.append(acc)
data=PDten307
resultI=[]    
for j in range(100):
    kf = KFold(n_splits=num_folds,shuffle=True)
    steps = []
    steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
    steps.append(('LR', LogisticRegression(multi_class='multinomial', solver='lbfgs')))
    model = Pipeline(steps)
    resultj = cross_val_score(model, data, Y, cv=kf)
    resultI.append(resultj.mean())
result = np.array(resultI)
acc = result.mean()
print('第%d次cv5准确率 : %s' % (j , result.mean()))
ACC.append(acc)
data=PDten309
resultI=[]    
for j in range(100):
    kf = KFold(n_splits=num_folds,shuffle=True)
    steps = []
    steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
    steps.append(('LR', LogisticRegression(multi_class='multinomial', solver='lbfgs')))
    model = Pipeline(steps)
    resultj = cross_val_score(model, data, Y, cv=kf)
    resultI.append(resultj.mean())
result = np.array(resultI)
acc = result.mean()
print('第%d次cv5准确率 : %s' % (j , result.mean()))
ACC.append(acc)

data=PDten501
resultI=[]    
for j in range(100):
    kf = KFold(n_splits=num_folds,shuffle=True)
    steps = []
    steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
    steps.append(('LR', LogisticRegression(multi_class='multinomial', solver='lbfgs')))
    model = Pipeline(steps)
    resultj = cross_val_score(model, data, Y, cv=kf)
    resultI.append(resultj.mean())
result = np.array(resultI)
acc = result.mean()
print('第%d次cv5准确率 : %s' % (j , result.mean()))
ACC.append(acc)
data=PDten503
resultI=[]    
for j in range(100):
    kf = KFold(n_splits=num_folds,shuffle=True)
    steps = []
    steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
    steps.append(('LR', LogisticRegression(multi_class='multinomial', solver='lbfgs')))
    model = Pipeline(steps)
    resultj = cross_val_score(model, data, Y, cv=kf)
    resultI.append(resultj.mean())
result = np.array(resultI)
acc = result.mean()
print('第%d次cv5准确率 : %s' % (j , result.mean()))
ACC.append(acc)
data=PDten505
resultI=[]    
for j in range(100):
    kf = KFold(n_splits=num_folds,shuffle=True)
    steps = []
    steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
    steps.append(('LR', LogisticRegression(multi_class='multinomial', solver='lbfgs')))
    model = Pipeline(steps)
    resultj = cross_val_score(model, data, Y, cv=kf)
    resultI.append(resultj.mean())
result = np.array(resultI)
acc = result.mean()
print('第%d次cv5准确率 : %s' % (j , result.mean()))
ACC.append(acc)
data=PDten507
resultI=[]    
for j in range(100):
    kf = KFold(n_splits=num_folds,shuffle=True)
    steps = []
    steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
    steps.append(('LR', LogisticRegression(multi_class='multinomial', solver='lbfgs')))
    model = Pipeline(steps)
    resultj = cross_val_score(model, data, Y, cv=kf)
    resultI.append(resultj.mean())
result = np.array(resultI)
acc = result.mean()
print('第%d次cv5准确率 : %s' % (j , result.mean()))
ACC.append(acc)
data=PDten509
resultI=[]    
for j in range(100):
    kf = KFold(n_splits=num_folds,shuffle=True)
    steps = []
    steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
    steps.append(('LR', LogisticRegression(multi_class='multinomial', solver='lbfgs')))
    model = Pipeline(steps)
    resultj = cross_val_score(model, data, Y, cv=kf)
    resultI.append(resultj.mean())
result = np.array(resultI)
acc = result.mean()
print('第%d次cv5准确率 : %s' % (j , result.mean()))
ACC.append(acc)

data=PDten701
resultI=[]    
for j in range(100):
    kf = KFold(n_splits=num_folds,shuffle=True)
    steps = []
    steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
    steps.append(('LR', LogisticRegression(multi_class='multinomial', solver='lbfgs')))
    model = Pipeline(steps)
    resultj = cross_val_score(model, data, Y, cv=kf)
    resultI.append(resultj.mean())
result = np.array(resultI)
acc = result.mean()
print('第%d次cv5准确率 : %s' % (j , result.mean()))
ACC.append(acc)
data=PDten703
resultI=[]    
for j in range(100):
    kf = KFold(n_splits=num_folds,shuffle=True)
    steps = []
    steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
    steps.append(('LR', LogisticRegression(multi_class='multinomial', solver='lbfgs')))
    model = Pipeline(steps)
    resultj = cross_val_score(model, data, Y, cv=kf)
    resultI.append(resultj.mean())
result = np.array(resultI)
acc = result.mean()
print('第%d次cv5准确率 : %s' % (j , result.mean()))
ACC.append(acc)
data=PDten705
resultI=[]    
for j in range(100):
    kf = KFold(n_splits=num_folds,shuffle=True)
    steps = []
    steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
    steps.append(('LR', LogisticRegression(multi_class='multinomial', solver='lbfgs')))
    model = Pipeline(steps)
    resultj = cross_val_score(model, data, Y, cv=kf)
    resultI.append(resultj.mean())
result = np.array(resultI)
acc = result.mean()
print('第%d次cv5准确率 : %s' % (j , result.mean()))
ACC.append(acc)
data=PDten707
resultI=[]    
for j in range(100):
    kf = KFold(n_splits=num_folds,shuffle=True)
    steps = []
    steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
    steps.append(('LR', LogisticRegression(multi_class='multinomial', solver='lbfgs')))
    model = Pipeline(steps)
    resultj = cross_val_score(model, data, Y, cv=kf)
    resultI.append(resultj.mean())
result = np.array(resultI)
acc = result.mean()
print('第%d次cv5准确率 : %s' % (j , result.mean()))
ACC.append(acc)
data=PDten709
resultI=[]    
for j in range(100):
    kf = KFold(n_splits=num_folds,shuffle=True)
    steps = []
    steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
    steps.append(('LR', LogisticRegression(multi_class='multinomial', solver='lbfgs')))
    model = Pipeline(steps)
    resultj = cross_val_score(model, data, Y, cv=kf)
    resultI.append(resultj.mean())
result = np.array(resultI)
acc = result.mean()
print('第%d次cv5准确率 : %s' % (j , result.mean()))
ACC.append(acc)

data=PDten901
resultI=[]    
for j in range(100):
    kf = KFold(n_splits=num_folds,shuffle=True)
    steps = []
    steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
    steps.append(('LR', LogisticRegression(multi_class='multinomial', solver='lbfgs')))
    model = Pipeline(steps)
    resultj = cross_val_score(model, data, Y, cv=kf)
    resultI.append(resultj.mean())
result = np.array(resultI)
acc = result.mean()
print('第%d次cv5准确率 : %s' % (j , result.mean()))
ACC.append(acc)
data=PDten903
resultI=[]    
for j in range(100):
    kf = KFold(n_splits=num_folds,shuffle=True)
    steps = []
    steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
    steps.append(('LR', LogisticRegression(multi_class='multinomial', solver='lbfgs')))
    model = Pipeline(steps)
    resultj = cross_val_score(model, data, Y, cv=kf)
    resultI.append(resultj.mean())
result = np.array(resultI)
acc = result.mean()
print('第%d次cv5准确率 : %s' % (j , result.mean()))
ACC.append(acc)
data=PDten905
resultI=[]    
for j in range(100):
    kf = KFold(n_splits=num_folds,shuffle=True)
    steps = []
    steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
    steps.append(('LR', LogisticRegression(multi_class='multinomial', solver='lbfgs')))
    model = Pipeline(steps)
    resultj = cross_val_score(model, data, Y, cv=kf)
    resultI.append(resultj.mean())
result = np.array(resultI)
acc = result.mean()
print('第%d次cv5准确率 : %s' % (j , result.mean()))
ACC.append(acc)
data=PDten907
resultI=[]    
for j in range(100):
    kf = KFold(n_splits=num_folds,shuffle=True)
    steps = []
    steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
    steps.append(('LR', LogisticRegression(multi_class='multinomial', solver='lbfgs')))
    model = Pipeline(steps)
    resultj = cross_val_score(model, data, Y, cv=kf)
    resultI.append(resultj.mean())
result = np.array(resultI)
acc = result.mean()
print('第%d次cv5准确率 : %s' % (j , result.mean()))
ACC.append(acc)
data=PDten909
resultI=[]    
for j in range(100):
    kf = KFold(n_splits=num_folds,shuffle=True)
    steps = []
    steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
    steps.append(('LR', LogisticRegression(multi_class='multinomial', solver='lbfgs')))
    model = Pipeline(steps)
    resultj = cross_val_score(model, data, Y, cv=kf)
    resultI.append(resultj.mean())
result = np.array(resultI)
acc = result.mean()
print('第%d次cv5准确率 : %s' % (j , result.mean()))
ACC.append(acc)

data=PDten1101
resultI=[]    
for j in range(100):
    kf = KFold(n_splits=num_folds,shuffle=True)
    steps = []
    steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
    steps.append(('LR', LogisticRegression(multi_class='multinomial', solver='lbfgs')))
    model = Pipeline(steps)
    resultj = cross_val_score(model, data, Y, cv=kf)
    resultI.append(resultj.mean())
result = np.array(resultI)
acc = result.mean()
print('第%d次cv5准确率 : %s' % (j , result.mean()))
ACC.append(acc)
data=PDten1103
resultI=[]    
for j in range(100):
    kf = KFold(n_splits=num_folds,shuffle=True)
    steps = []
    steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
    steps.append(('LR', LogisticRegression(multi_class='multinomial', solver='lbfgs')))
    model = Pipeline(steps)
    resultj = cross_val_score(model, data, Y, cv=kf)
    resultI.append(resultj.mean())
result = np.array(resultI)
acc = result.mean()
print('第%d次cv5准确率 : %s' % (j , result.mean()))
ACC.append(acc)
data=PDten1105
resultI=[]    
for j in range(100):
    kf = KFold(n_splits=num_folds,shuffle=True)
    steps = []
    steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
    steps.append(('LR', LogisticRegression(multi_class='multinomial', solver='lbfgs')))
    model = Pipeline(steps)
    resultj = cross_val_score(model, data, Y, cv=kf)
    resultI.append(resultj.mean())
result = np.array(resultI)
acc = result.mean()
print('第%d次cv5准确率 : %s' % (j , result.mean()))
ACC.append(acc)
data=PDten1107
resultI=[]    
for j in range(100):
    kf = KFold(n_splits=num_folds,shuffle=True)
    steps = []
    steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
    steps.append(('LR', LogisticRegression(multi_class='multinomial', solver='lbfgs')))
    model = Pipeline(steps)
    resultj = cross_val_score(model, data, Y, cv=kf)
    resultI.append(resultj.mean())
result = np.array(resultI)
acc = result.mean()
print('第%d次cv5准确率 : %s' % (j , result.mean()))
ACC.append(acc)
data=PDten1109
resultI=[]    
for j in range(100):
    kf = KFold(n_splits=num_folds,shuffle=True)
    steps = []
    steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
    steps.append(('LR', LogisticRegression(multi_class='multinomial', solver='lbfgs')))
    model = Pipeline(steps)
    resultj = cross_val_score(model, data, Y, cv=kf)
    resultI.append(resultj.mean())
result = np.array(resultI)
acc = result.mean()
print('第%d次cv5准确率 : %s' % (j , result.mean()))
ACC.append(acc)

data=PDten1301
resultI=[]    
for j in range(100):
    kf = KFold(n_splits=num_folds,shuffle=True)
    steps = []
    steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
    steps.append(('LR', LogisticRegression(multi_class='multinomial', solver='lbfgs')))
    model = Pipeline(steps)
    resultj = cross_val_score(model, data, Y, cv=kf)
    resultI.append(resultj.mean())
result = np.array(resultI)
acc = result.mean()
print('第%d次cv5准确率 : %s' % (j , result.mean()))
ACC.append(acc)
data=PDten1303
resultI=[]    
for j in range(100):
    kf = KFold(n_splits=num_folds,shuffle=True)
    steps = []
    steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
    steps.append(('LR', LogisticRegression(multi_class='multinomial', solver='lbfgs')))
    model = Pipeline(steps)
    resultj = cross_val_score(model, data, Y, cv=kf)
    resultI.append(resultj.mean())
result = np.array(resultI)
acc = result.mean()
print('第%d次cv5准确率 : %s' % (j , result.mean()))
ACC.append(acc)
data=PDten1305
resultI=[]    
for j in range(100):
    kf = KFold(n_splits=num_folds,shuffle=True)
    steps = []
    steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
    steps.append(('LR', LogisticRegression(multi_class='multinomial', solver='lbfgs')))
    model = Pipeline(steps)
    resultj = cross_val_score(model, data, Y, cv=kf)
    resultI.append(resultj.mean())
result = np.array(resultI)
acc = result.mean()
print('第%d次cv5准确率 : %s' % (j , result.mean()))
ACC.append(acc)
data=PDten1307
resultI=[]    
for j in range(100):
    kf = KFold(n_splits=num_folds,shuffle=True)
    steps = []
    steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
    steps.append(('LR', LogisticRegression(multi_class='multinomial', solver='lbfgs')))
    model = Pipeline(steps)
    resultj = cross_val_score(model, data, Y, cv=kf)
    resultI.append(resultj.mean())
result = np.array(resultI)
acc = result.mean()
print('第%d次cv5准确率 : %s' % (j , result.mean()))
ACC.append(acc)
data=PDten1309
resultI=[]    
for j in range(100):
    kf = KFold(n_splits=num_folds,shuffle=True)
    steps = []
    steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
    steps.append(('LR', LogisticRegression(multi_class='multinomial', solver='lbfgs')))
    model = Pipeline(steps)
    resultj = cross_val_score(model, data, Y, cv=kf)
    resultI.append(resultj.mean())
result = np.array(resultI)
acc = result.mean()
print('第%d次cv5准确率 : %s' % (j , result.mean()))
ACC.append(acc)

data=PDten1501
resultI=[]    
for j in range(100):
    kf = KFold(n_splits=num_folds,shuffle=True)
    steps = []
    steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
    steps.append(('LR', LogisticRegression(multi_class='multinomial', solver='lbfgs')))
    model = Pipeline(steps)
    resultj = cross_val_score(model, data, Y, cv=kf)
    resultI.append(resultj.mean())
result = np.array(resultI)
acc = result.mean()
print('第%d次cv5准确率 : %s' % (j , result.mean()))
ACC.append(acc)
data=PDten1503
resultI=[]    
for j in range(100):
    kf = KFold(n_splits=num_folds,shuffle=True)
    steps = []
    steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
    steps.append(('LR', LogisticRegression(multi_class='multinomial', solver='lbfgs')))
    model = Pipeline(steps)
    resultj = cross_val_score(model, data, Y, cv=kf)
    resultI.append(resultj.mean())
result = np.array(resultI)
acc = result.mean()
print('第%d次cv5准确率 : %s' % (j , result.mean()))
ACC.append(acc)
data=PDten1505
resultI=[]    
for j in range(100):
    kf = KFold(n_splits=num_folds,shuffle=True)
    steps = []
    steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
    steps.append(('LR', LogisticRegression(multi_class='multinomial', solver='lbfgs')))
    model = Pipeline(steps)
    resultj = cross_val_score(model, data, Y, cv=kf)
    resultI.append(resultj.mean())
result = np.array(resultI)
acc = result.mean()
print('第%d次cv5准确率 : %s' % (j , result.mean()))
ACC.append(acc)
data=PDten1507
resultI=[]    
for j in range(100):
    kf = KFold(n_splits=num_folds,shuffle=True)
    steps = []
    steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
    steps.append(('LR', LogisticRegression(multi_class='multinomial', solver='lbfgs')))
    model = Pipeline(steps)
    resultj = cross_val_score(model, data, Y, cv=kf)
    resultI.append(resultj.mean())
result = np.array(resultI)
acc = result.mean()
print('第%d次cv5准确率 : %s' % (j , result.mean()))
ACC.append(acc)
data=PDten1509
resultI=[]    
for j in range(100):
    kf = KFold(n_splits=num_folds,shuffle=True)
    steps = []
    steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
    steps.append(('LR', LogisticRegression(multi_class='multinomial', solver='lbfgs')))
    model = Pipeline(steps)
    resultj = cross_val_score(model, data, Y, cv=kf)
    resultI.append(resultj.mean())
result = np.array(resultI)
acc = result.mean()
print('第%d次cv5准确率 : %s' % (j , result.mean()))
ACC.append(acc)

ACC = np.array(ACC)  

import pickle
output = open('PD40655_acc.pkl','wb')
pickle.dump(ACC,output)
output.close()

import pickle
pkl_file=open('PD40655_acc.pkl','rb')
ACC = pickle.load(pkl_file)
pkl_file.close()
