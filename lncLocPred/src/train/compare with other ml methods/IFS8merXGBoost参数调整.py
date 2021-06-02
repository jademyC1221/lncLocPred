# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 19:37:51 2019

@author: Chenmeijun
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import model_selection, metrics   
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt

import warnings
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

from scipy.io import loadmat
datadict = loadmat("lnc8mernor655CL.mat")
data = datadict.get('lnc8mernor655CL')   
labelxgbdict = loadmat("labelxgb.mat")  
label = labelxgbdict.get('labelxgb') 

data=np.array(data)
label=np.array(label)

#data=data[:,0:7645]
X=data
Y=label
import numpy as np
data_x = data
data_y = label
print(np.hstack((data_x, data_y)))
indices = np.random.permutation(data_x.shape[0])

rand_data_x = data_x[indices]
rand_data_y = data_y[indices]
print(np.hstack((rand_data_x, rand_data_y)))

train_x = rand_data_x[0:455,:]
train_y = rand_data_y[0:455,:]
test_x = rand_data_x[455:655,:]
test_y = rand_data_y[455:655,:]
#train_Y = np.squeeze(train_y)
c, r = train_y.shape
train_y = train_y.reshape(c,)
#dtrain=xgb.DMatrix(train_x, label=train_y)
#dtest=xgb.DMatrix(test_x,label=test_y)

#第一次：决策树的最佳数量也就是估计器的数目
# Fitting 5 folds for each of 20 candidates,totalling 50 fits
#if __name__ == "__main__":
cv_params = {'n_estimators': list(range(50,1050,50))}
other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1,
                    'objective': 'multi:softmax','num_class':4,'nthread':4}
model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='accuracy', cv=5, verbose=1)
#    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='accuracy', cv=5, verbose=1, n_jobs=4)

optimized_GBM.fit(train_x, train_y)
evalute_result = optimized_GBM.grid_scores_
print('每轮迭代运行结果:{0}'.format(evalute_result))
print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
#参数的最佳取值：{'n_estimators': 50}
#最佳模型得分:0.6725274725274726


#第二次
cv_params = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight': [1, 2, 3, 4, 5, 6]}
other_params = {'learning_rate': 0.1, 'n_estimators': 50, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1,
                    'objective': 'multi:softmax','num_class':4,'nthread':4}
model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='accuracy', cv=5, verbose=1)
optimized_GBM.fit(train_x, train_y)
evalute_result = optimized_GBM.grid_scores_
print('每轮迭代运行结果:{0}'.format(evalute_result))
print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
#参数的最佳取值：{'max_depth': 7, 'min_child_weight': 4}
#最佳模型得分:0.6857142857142857


#第三次
cv_params = {'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
other_params = {'learning_rate': 0.1, 'n_estimators': 50, 'max_depth': 7, 'min_child_weight': 4, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1,
                    'objective': 'multi:softmax','num_class':4,'nthread':4}
model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='accuracy', cv=5, verbose=1)
optimized_GBM.fit(train_x, train_y)
evalute_result = optimized_GBM.grid_scores_
print('每轮迭代运行结果:{0}'.format(evalute_result))
print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
#参数的最佳取值：{'gamma': 0.5}
#最佳模型得分:0.6681318681318681


#第四次
cv_params = {'subsample': [0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9]}
other_params = {'learning_rate': 0.1, 'n_estimators':50, 'max_depth': 7, 'min_child_weight':4, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.5, 'reg_alpha': 0, 'reg_lambda': 1,
                    'objective': 'multi:softmax','num_class':4,'nthread':4}
model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='accuracy', cv=5, verbose=1)
optimized_GBM.fit(train_x, train_y)
evalute_result = optimized_GBM.grid_scores_
print('每轮迭代运行结果:{0}'.format(evalute_result))
print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
#参数的最佳取值：{'colsample_bytree': 0.7, 'subsample': 0.9}
#最佳模型得分:0.6747252747252748


#第五次
cv_params = {'reg_alpha': [0.05, 0.1, 1, 2, 3], 'reg_lambda': [0.05, 0.1, 1, 2, 3]}
other_params = {'learning_rate': 0.1, 'n_estimators': 50, 'max_depth': 7, 'min_child_weight': 4, 'seed': 0,
                    'subsample': 0.9, 'colsample_bytree': 0.7, 'gamma': 0.5, 'reg_alpha': 0, 'reg_lambda': 1,
                    'objective': 'multi:softmax','num_class':4,'nthread':4}
model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='accuracy', cv=5, verbose=1)
optimized_GBM.fit(train_x, train_y)
evalute_result = optimized_GBM.grid_scores_
print('每轮迭代运行结果:{0}'.format(evalute_result))
print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
#参数的最佳取值：{'reg_alpha': 0.1, 'reg_lambda': 0.05}
#最佳模型得分:0.6681318681318681


#第六次
cv_params = {'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2]}
other_params = {'learning_rate': 0.1, 'n_estimators': 50, 'max_depth': 7, 'min_child_weight': 4, 'seed': 0,
                    'subsample': 0.9, 'colsample_bytree': 0.7, 'gamma': 0.5, 'reg_alpha': 0.1, 'reg_lambda': 0.05,
                     'objective': 'multi:softmax','num_class':4,'nthread':4}
model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='accuracy', cv=5, verbose=1)
optimized_GBM.fit(train_x, train_y)
evalute_result = optimized_GBM.grid_scores_
print('每轮迭代运行结果:{0}'.format(evalute_result))
print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
#参数的最佳取值：{'learning_rate': 0.1}
#最佳模型得分:0.6681318681318681


params={
        'booster':'gbtree',
        'objective': 'multi:softmax', #指明是分类问题
       # 'eval_metric': 'auc',
        'num_class':4, # 类数，与 multisoftmax 并用
        'gamma':0.5,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
        'max_depth':7, # 构建树的深度，越大越容易过拟合
        'lambda':0.05,  #控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        'subsample':0.9, # 采样训练数据，设置为0.5，随机选择一般的数据实例 (0:1]
        'colsample_bytree':0.7, # 构建树树时的采样比率 (0:1]
        'min_child_weight':4, # 节点的最少特征数
        'silent':0 ,# 设置成1则没有运行信息输出，最好是设置为0.
        'eta': 0.1, # 如同学习率
        'seed':710,
        'alpha':0.1,
        'nthread':4,# cpu 线程数,根据自己U的个数适当调整
        'n_estimators': 50,
}
other_params = {'learning_rate': 0.1, 'n_estimators': 50, 'max_depth': 7, 
                'min_child_weight': 4, 'seed': 0,'subsample': 0.9, 
                'colsample_bytree': 0.7, 'gamma': 0.5, 'reg_alpha': 0.1, 
                'reg_lambda': 0.05,'objective': 'multi:softmax','num_class':4,'nthread':4}


