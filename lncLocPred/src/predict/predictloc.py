# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 12:20:35 2020

@author: chenmeijun
"""

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import  Pipeline
import numpy as np
from sklearn.model_selection import GridSearchCV
from scipy.io import loadmat
import os
import re


def predictloc():

    father_path = os.getcwd()
    psedncfilename = os.getcwd()+'\datapse.txt'
    tripletfilename = os.getcwd()+'\datatriplet.txt'
    
    kmernor8dict = loadmat(os.getcwd()+"\kmernor.mat")  
    kmernor8 = kmernor8dict.get('kmernor')    
    kmernorclorderdict = loadmat(os.getcwd()+"\kmernorclorder.mat")
    kmernorclorder8 = kmernorclorderdict.get('kmernorclorder') 
    
    # kmerfeature8 = kmernorclorder8[:,0:9080]
    
    #kmernorcl9080
    datadict = loadmat(os.getcwd()+"\kmernorcl9080.mat")
    kmerfeature8 = datadict.get('kmernorcl9080') 
    
    
    datapse = []
    for line in open(psedncfilename,"r"):  
            line=line.strip().split("\t")
            result=[]
            for i in line:
                digi_str=re.findall('^[\\+\\-]?[\\d]+(\\.[\\d]+)?$',i)
                if len(digi_str)!=0:
                    i=float(i)
                    result.append(i)
            if len(result)!=0:
                datapse.append(result)
    datatriplet=[]
    for line in open(tripletfilename,"r"):  
            line=line.strip().split("\t")
            result=[]
            for i in line:
                digi_str=re.findall('^[\\+\\-]?[\\d]+(\\.[\\d]+)?$',i)
                if len(digi_str)!=0:
                    i=float(i)
                    result.append(i)
            if len(result)!=0:
                datatriplet.append(result)
    datatriplet=np.array(datatriplet)
    datapse=np.array(datapse)
     
    import pickle
    pkl_file=open('lncLocPredtp8.pkl','rb')    
    traindata = pickle.load(pkl_file)
    trainlabel = pickle.load(pkl_file)
    Fscoretp8 = pickle.load(pkl_file)
    pkl_file.close()
    
    
    testdata1 = np.hstack((datatriplet,datapse,kmerfeature8))
    testdata2=list()
    Fscoretp8=Fscoretp8-1
    for j in range(0,len(Fscoretp8),1):
        a = testdata1[:,Fscoretp8[j]]
        testdata2.append(a)
    testdata2=np.array(testdata2).T
    testdata3=testdata2.reshape(len(testdata2[0]),len(testdata2[0][0]))
    testdata = testdata3[:,0:7228]
    

    steps = []
    steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
    steps.append(('LR', LogisticRegression(C =1000, multi_class='multinomial', solver='lbfgs')))
    model = Pipeline(steps)
    
    trainlabel = trainlabel.reshape(len(trainlabel),)
    clf = model.fit(traindata, trainlabel)
    
    #location : 1:Nucleus ; 2:Cytoplasm ; 3:Ribosome ; 4:Exosome 
    location = clf.predict(testdata) 
    #p : Probability of predicted sequence at various subcellular locations
    p = clf.predict_proba(testdata)

    locationname=[]
    for i in range(len(location)):
        if location[i]==1:
            locationname.append('Nucleus')
        if location[i]==2:
            locationname.append('Cytoplam')
        if location[i]==3:
            locationname.append('Ribosome')
        if location[i]==4:
            locationname.append('Exosome')
    P=[]
    for i in range(len(location)):
        x=location[i]-1
        P.append(p[i,x])
    SequenceOrder=[]
    for i in range(len(location)):
        SequenceOrder.append(i+1)
        
    resultt = list(zip(SequenceOrder,locationname,P))
    
    output = open(father_path+'\locationResult.txt','w',encoding='gbk')
    output.write('SequenceOrder,\t Location,\t Probability\n')
    for row in resultt:	
        rowtxt = '{},\t {},\t {}'.format(row[0],row[1],row[2])	
        output.write(rowtxt)	
        output.write('\n')
    output.close()
    
    print('End of prediction: The prediction result `` locationResult.txt '' has been saved in the current folder.')

if __name__ == "__main__":
    predictloc()