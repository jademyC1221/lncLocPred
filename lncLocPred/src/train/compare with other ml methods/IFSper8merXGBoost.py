from scipy.io import loadmat
datadict = loadmat("lnc8mernor655CL_2.mat")
data = datadict.get('lnc8mernor655CL_2')   
labelxgbdict = loadmat("labelxgb.mat")  
label = labelxgbdict.get('labelxgb') 
Y=label

#每个特征集都进行5次5折交叉验证,直接填入参数
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import math
percent = [x/100 for x in range(2,102,2)]
Num=[]
for i in range(len(percent)):
    Num.append(math.ceil(64855*percent[i]))
#other_params = {'learning_rate': 0.1, 'n_estimators': 50, 'max_depth': 7, 'min_child_weight': 4, 'seed': 0,
#                    'subsample': 0.9, 'colsample_bytree': 0.7, 'gamma': 0.5, 'reg_alpha': 0.1, 'reg_lambda': 0.05,
#                     'objective': 'multi:softmax','num_class':4,'nthread':4}

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
        model = xgb.XGBClassifier(objective= 'multi:softmax',num_class=4,learning_rate= 0.1, 
                                  n_estimators=50, max_depth=7, min_child_weight= 4,
                                  subsample=0.9, colsample_bytree=0.7, gamma=0.5, nthread=4)
        resultj = cross_val_score(model, rescaledX, Y, cv=kf)
        resultI.append(resultj.mean())
    acci = np.array(resultI)
    print('加入第%d个准确率 : %s' % (i , acci.mean()))
    OUT1.append(Num[i])
    OUT2.append(acci.mean())

ifs8merout1=OUT1
ifs8merout2=OUT2
#合并
IFS00 = np.vstack((OUT1,OUT2)).T
#IFS00 = np.vstack((IFS001,IFS002))
IFS01 =np.argsort(-IFS00[:,1])
IFSorder0 = IFS00[IFS01].tolist()
Xgbifs8perorder0 = IFSorder0

#保存数据
import pickle
output = open('XgbLRIFSpercent655.pkl','wb')
pickle.dump(data,output)
pickle.dump(label,output)
pickle.dump(ifs8merout1,output)
pickle.dump(ifs8merout2,output)
pickle.dump(IFS00,output)
pickle.dump(Xgbifs8perorder0,output)
output.close()

#读取数据
import pickle
pkl_file=open('XgbLRIFSpercent655.pkl','rb')
data=pickle.load(pkl_file)
label=pickle.load(pkl_file)
ifs8merout1=pickle.load(pkl_file)
ifs8merout2=pickle.load(pkl_file)
IFS00=pickle.load(pkl_file)
Xgbifs8perorder0=pickle.load(pkl_file)
pkl_file.close()


'''然后和triplet，psednc特征结合'''



from scipy.io import loadmat
datadict = loadmat("pertp8XGBfsc.mat")  
data = datadict.get('pertp8XGBfsc')     
labeldict = loadmat("label.mat")  
label = labeldict.get('label') 
Y=label
#每个特征集都进行5次5折交叉验证,直接填入参数
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import numpy as np
import math
from sklearn.model_selection import GridSearchCV

X=data
Y=label
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
#参数的最佳取值：{'n_estimators': 100}


#第二次
cv_params = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight': [1, 2, 3, 4, 5, 6]}
other_params = {'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1,
                    'objective': 'multi:softmax','num_class':4,'nthread':4}
model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='accuracy', cv=5, verbose=1)
optimized_GBM.fit(train_x, train_y)
evalute_result = optimized_GBM.grid_scores_
print('每轮迭代运行结果:{0}'.format(evalute_result))
print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
#参数的最佳取值：{'max_depth': 4, 'min_child_weight': 3}


#第三次
cv_params = {'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
other_params = {'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 4, 'min_child_weight': 3, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1,
                    'objective': 'multi:softmax','num_class':4,'nthread':4}
model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='accuracy', cv=5, verbose=1)
optimized_GBM.fit(train_x, train_y)
evalute_result = optimized_GBM.grid_scores_
print('每轮迭代运行结果:{0}'.format(evalute_result))
print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
#参数的最佳取值：{'gamma': 0.3}
#最佳模型得分:0.7076923076923077


#第四次
cv_params = {'subsample': [0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9]}
other_params = {'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 4, 'min_child_weight': 3, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.3, 'reg_alpha': 0, 'reg_lambda': 1,
                    'objective': 'multi:softmax','num_class':4,'nthread':4}
model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='accuracy', cv=5, verbose=1)
optimized_GBM.fit(train_x, train_y)
evalute_result = optimized_GBM.grid_scores_
print('每轮迭代运行结果:{0}'.format(evalute_result))
print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
#参数的最佳取值：{'colsample_bytree': 0.6, 'subsample': 0.7}
#最佳模型得分:0.7098901098901099


#第五次
cv_params = {'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2]}
other_params = {'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 4, 'min_child_weight': 3, 'seed': 0,
                    'subsample': 0.7, 'colsample_bytree': 0.6, 'gamma': 0.3, 'reg_alpha': 0, 'reg_lambda': 1,
                    'objective': 'multi:softmax','num_class':4,'nthread':4}
model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='accuracy', cv=5, verbose=1)
optimized_GBM.fit(train_x, train_y)
evalute_result = optimized_GBM.grid_scores_
print('每轮迭代运行结果:{0}'.format(evalute_result))
print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))



percent = [x/100 for x in range(1,101,1)]
Num=[]
for i in range(len(percent)):
    Num.append(math.ceil(2846*percent[i]))
    
OUT1=list()
OUT2=list()
num_folds = 5
for i in range(len(percent)):
    X=data[:, 0:Num[i]]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    rescaledX = scaler.fit_transform(X)   
    resultI=[]    
    for j in range(5):
        kf = KFold(n_splits=num_folds,shuffle=True)
        model = xgb.XGBClassifier(learning_rate =0.1,booster='gbtree',n_estimators=100,
                                  max_depth=4,min_child_weight=3,gamma=0.3,subsample=0.7,
                                  colsample_bytree=0.6,objective= 'multi:softmax',
                                  num_class=4,nthread=3)
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
ifspertp8XGBorder0 = IFSorder0

#保存数据
import pickle
output = open('IFSpertp8XGB.pkl','wb')
pickle.dump(data,output)
pickle.dump(label,output)
pickle.dump(ifspertp8out1,output)
pickle.dump(ifspertp8out2,output)
pickle.dump(ifspertp8XGBorder0,output)
output.close()

#读取数据
import pickle
pkl_file=open('IFSpertp8XGB.pkl','rb')
data=pickle.load(pkl_file)
label=pickle.load(pkl_file)
ifspertp8out1=pickle.load(pkl_file)
ifspertp8out2=pickle.load(pkl_file)
pickle.dump(ifspertp8XGBorder0,output)
pkl_file.close()

#留一法
from sklearn.pipeline import  Pipeline
from sklearn.model_selection import LeaveOneOut
X=data
Y=label
data=X[:,0:1025]

loocv = LeaveOneOut()
steps = []
steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))

'''需要修改参数'''
steps.append(('XGB', xgb.XGBClassifier(learning_rate =0.1,booster='gbtree',n_estimators=100,
                                  max_depth=4,min_child_weight=3,gamma=0.3,subsample=0.7,
                                  colsample_bytree=0.6,objective= 'multi:softmax',
                                  num_class=4,nthread=3)))

model = Pipeline(steps)
result = cross_val_score(model, data, Y, cv=loocv)
print(result.mean())
acc=result.mean()
print("算法评估结果：%.7f%% (%.7f%%)" % (result.mean() * 100, result.std() * 100))
loocv.get_n_splits(data)
a=loocv._iter_test_indices(data)
prediction=[]
for train_index, test_index in loocv.split(data):
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    model.fit(X_train,y_train)
    predictioni=model.predict(X_test)
    prediction.append(predictioni)
Prediction=np.array(prediction)  

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
output = open('IFSpertp8XGBloocv.pkl','wb')
pickle.dump(X,output)
pickle.dump(Y,output)
pickle.dump(result,output)
pickle.dump(acc,output)
pickle.dump(Prediction,output)
pickle.dump(SN,output)
pickle.dump(SP,output)
pickle.dump(MCC,output)
output.close()
#读取数据
import pickle
pkl_file=open('IFSpertp8XGBloocv.pkl','rb')
X=pickle.load(pkl_file)
Y=pickle.load(pkl_file)
result=pickle.load(pkl_file)
acc=pickle.load(pkl_file)
Prediction=pickle.load(pkl_file)
SN=pickle.load(pkl_file)
SP=pickle.load(pkl_file)
MCC=pickle.load(pkl_file)
pkl_file.close()
