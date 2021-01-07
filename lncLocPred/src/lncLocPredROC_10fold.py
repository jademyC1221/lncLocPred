# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 09:29:26 2020

@author: chenmeijun
"""

from scipy.io import loadmat
datadict = loadmat("pertp8fsc.mat")  
X = datadict.get('pertp8fsc')  
data=X[:,0:7228]   
labeldict = loadmat("label.mat")  
label = labeldict.get('label') 
Y=label

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import  Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
# random_state = np.random.RandomState(0)
# X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
#kf = KFold(n_splits=num_folds,shuffle=True)

y_bin = label_binarize(Y, classes=[1, 2, 3, 4])
n_classes = y_bin.shape[1]
n_samples, n_features = data.shape

kf = KFold(n_splits=10,shuffle=True)
steps = []
steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
steps.append(('LR', LogisticRegression(C =1000, multi_class='multinomial', solver='lbfgs')))
model = Pipeline(steps)
classifier = OneVsRestClassifier(Pipeline(steps))
# result = cross_val_score(model, data, Y, cv=kf)
# print(result.mean())
# acc=result.mean()

kf=KFold(n_splits=10,shuffle=True)
from sklearn.model_selection import cross_val_predict
Y=Y.reshape(655,)
y_score = cross_val_predict(classifier, data, Y, cv=kf, method='predict_proba')

# y_score = cross_val_predict(classifier, data, Y ,cv=10,method='predict_proba')

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二微观）
fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# Compute macro-average ROC curve and ROC area（方法一宏观）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)

for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])  
# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

#保存数据
import pickle
output = open('LncLocPredtp8ROC10fold.pkl','wb')
pickle.dump(X,output)
pickle.dump(Y,output)
pickle.dump(data,output)
pickle.dump(y_score,output)
pickle.dump(y_bin,output)
pickle.dump(fpr,output)
pickle.dump(tpr,output)
pickle.dump(roc_auc,output)
output.close()
#读取数据
import pickle
pkl_file=open('LncLocPredtp8ROC10fold.pkl','rb')
X=pickle.load(pkl_file)
Y=pickle.load(pkl_file)
data=pickle.load(pkl_file)
y_score=pickle.load(pkl_file)
y_bin=pickle.load(pkl_file)
fpr=pickle.load(pkl_file)
tpr=pickle.load(pkl_file)
roc_auc=pickle.load(pkl_file)
pkl_file.close()
 


lw=1
plt.figure()

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 4.2, 
}
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 11,
}

# Plot all ROC curves
# lw=2
plt.figure()

plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.3f})'
               ''.format(roc_auc["micro"]),
         color='deeppink',  linestyle='--',linewidth=2)                  

#linestyle=':',用fpr和tpr，roc_auc的值画ROC，此处画的是总体的ROC曲线
plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.3f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle='--', linewidth=2)

#Every class ROC
colors = ['green', 'dimgrey', 'red','blue']
# colors = ['blue', 'red', 'green','black']
# colors = ['deeppink', 'aqua', 'cornflowerblue', 'green'],color='red',color='darkorange'
labeli = ['ROC curve of class nucleus', 'ROC curve of class cytoplasm', 'ROC curve of class ribosome', 'ROC curve of class exosome']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label= labeli[i]+' (area = {1:0.3f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=1.7)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

plt.xticks(fontsize=9,fontname = 'Times New Roman')
plt.yticks(fontsize=9,fontname = 'Times New Roman')
# plt.tick_params(labelsize=10)  
# labels = ax.get_xticklabels() + ax.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]

plt.xlabel('False Positive Rate',font2)
plt.ylabel('True Positive Rate',font2)
#plt.title('ROC curve of different locations')
plt.legend(loc="lower right",prop=font1)
plt.rcParams['savefig.dpi'] = 600 #图片像素
plt.rcParams['figure.dpi'] = 600 #分辨率
#plt.legend()
plt.savefig('lncLocPredtp8roc5.png', dpi=600) #指定分辨率保存
plt.show()

