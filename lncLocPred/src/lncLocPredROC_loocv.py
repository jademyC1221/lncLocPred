# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 18:03:42 2021

@author: chenmeijun
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

from scipy.io import loadmat
datadict = loadmat("pertp8fsc.mat")  
X = datadict.get('pertp8fsc')     
labeldict = loadmat("label.mat")  
label = labeldict.get('label') 
Y=label
data=X[:,0:7228]

# 将标签二值化
y = label_binarize(Y, classes=[1, 2, 3, 4])
# 设置种类
n_classes = y.shape[1]

# 训练模型并预测
n_samples, n_features = data.shape

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import  Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut

loocv = LeaveOneOut()
steps = []
steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
steps.append(('LR', LogisticRegression(C =1000, multi_class='multinomial', solver='lbfgs')))
classifier = OneVsRestClassifier(Pipeline(steps))

Y_score1=[]
Y_test1=[]
for train_index, test_index in loocv.split(data):
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = y[train_index], y[test_index]
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    Y_score1.append(y_score)
    Y_test1.append(y_test)
Y_score1=np.array(Y_score1)
Y_test1=np.array(Y_test1)

Yscore1=Y_score1[:,0]
Ytest1=Y_test1[:,0]


fpr1 = dict()
tpr1 = dict()
roc_auc1 = dict()

# 计算每一类的ROC
for i in range(n_classes):
    fpr1[i], tpr1[i], _ = roc_curve(Ytest1[:, i], Yscore1[:, i])
    roc_auc1[i] = auc(fpr1[i], tpr1[i])


# Compute micro-average ROC curve and ROC area（方法二微观）
fpr1["micro"], tpr1["micro"], _ = roc_curve(Ytest1.ravel(), Yscore1.ravel())
roc_auc1["micro"] = auc(fpr1["micro"], tpr1["micro"])

# Compute macro-average ROC curve and ROC area（方法一宏观）
# First aggregate all false positive rates
all_fpr1 = np.unique(np.concatenate([fpr1[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr1 = np.zeros_like(all_fpr1)

for i in range(n_classes):
    mean_tpr1 += interp(all_fpr1, fpr1[i], tpr1[i])
    
# Finally average it and compute AUC
mean_tpr1 /= n_classes
fpr1["macro"] = all_fpr1
tpr1["macro"] = mean_tpr1
roc_auc1["macro"] = auc(fpr1["macro"], tpr1["macro"])


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

plt.plot(fpr1["micro"], tpr1["micro"],
         label='micro-average ROC curve (area = {0:0.3f})'
               ''.format(roc_auc1["micro"]),
         color='deeppink',  linestyle='--',linewidth=2)                  

#linestyle=':',用fpr和tpr，roc_auc的值画ROC，此处画的是总体的ROC曲线
plt.plot(fpr1["macro"], tpr1["macro"],
         label='macro-average ROC curve (area = {0:0.3f})'
               ''.format(roc_auc1["macro"]),
         color='navy', linestyle='--', linewidth=2)

#Every class ROC
colors = ['green', 'dimgrey', 'red','blue']
# colors = ['blue', 'red', 'green','black']
# colors = ['deeppink', 'aqua', 'cornflowerblue', 'green'],color='red',color='darkorange'
labeli = ['ROC curve of class nucleus', 'ROC curve of class cytoplasm', 'ROC curve of class ribosome', 'ROC curve of class exosome']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr1[i], tpr1[i], color=color, lw=2,
             label= labeli[i]+' (area = {1:0.3f})'
             ''.format(i, roc_auc1[i]))

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
plt.savefig('lncLocPredtp8roc_LOOCV_.png', dpi=600) #指定分辨率保存
plt.show()


#保存数据
import pickle
output = open('pertp8fscRoc7228_.pkl','wb')
pickle.dump(X,output)
pickle.dump(Y,output)
pickle.dump(data,output)
pickle.dump(Y_score1,output)
pickle.dump(Y_test1,output)
pickle.dump(Yscore1,output)
pickle.dump(Ytest1,output)
pickle.dump(fpr1,output)
pickle.dump(tpr1,output)
pickle.dump(roc_auc1,output)
output.close()
#读取数据
import pickle
pkl_file=open('pertp8fscRoc7228_.pkl','rb')    
X=pickle.load(pkl_file)
Y=pickle.load(pkl_file)
data=pickle.load(pkl_file)
Y_score1=pickle.load(pkl_file)
Y_test1=pickle.load(pkl_file)
Yscore1=pickle.load(pkl_file)
Ytest1=pickle.load(pkl_file)
fpr1=pickle.load(pkl_file)
tpr1=pickle.load(pkl_file)
roc_auc1=pickle.load(pkl_file)
pkl_file.close()
