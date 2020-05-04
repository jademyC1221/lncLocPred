from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
import math
from sklearn.model_selection import GridSearchCV
from scipy.io import loadmat
#pertp8fsc.mat is obtained from Fscore.m
datadict = loadmat("pertp8fsc.mat")  
data = datadict.get('pertp8fsc')    
labeldict = loadmat("label.mat")
label = labeldict.get('label') 
Y=label

percent = [x/100 for x in range(1,101,1)]
Num=[]
for i in range(len(percent)):
    Num.append(math.ceil(10628*percent[i]))
    
ifspertp8out1=list()
ifspertp8out2=list()
ifspertp8out3=list()
num_folds = 5
for i in range(len(percent)):
    X=data[:, 0:Num[i]]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    rescaledX = scaler.fit_transform(X)
    
    resultIgs=[]
    resultIparams=[]
    for x in range(3):
        num_folds = 5
        kf = KFold(n_splits=num_folds,shuffle=True)
        param_grid = {}
        param_grid['C'] = [100, 1000]
#        param_grid['solver'] = ['newton-cg','lbfgs']
#        param_grid['multi_class'] = ['ovr','multinomial']
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=kf)
        grid_result = grid.fit(rescaledX,label)
        print('The highest acc : %s parameters %s' % (grid_result.best_score_, grid_result.best_params_))
        resultIgs.append(grid_result.best_score_)
        resultIparams.append(grid_result.best_params_)
    IFS00 = list(zip(resultIgs,resultIparams))
    IFS01 = np.argsort(-np.array(resultIgs))
    params = resultIparams[IFS01[0]]
    
    resultI=[]    
    for j in range(5):
        kf = KFold(n_splits=num_folds,shuffle=True)
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs',**params)
        resultj = cross_val_score(model, rescaledX, Y, cv=kf)
        resultI.append(resultj.mean())
    acci = np.array(resultI)
    print('Add the number of %d accuracy: %s' % (i , acci.mean()))
    ifspertp8out1.append(Num[i])
    ifspertp8out2.append(params)
    ifspertp8out3.append(acci.mean())

IFS00 = np.vstack((ifspertp8out1,ifspertp8out2,ifspertp8out3)).T
IFS01 =np.argsort(-IFS00[:,2])
IFSorder0 = IFS00[IFS01].tolist()
ifspertp8order0 = IFSorder0
