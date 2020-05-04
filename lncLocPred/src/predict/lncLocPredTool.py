# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 16:59:58 2020

@author: CMJ
"""

#import tkinter
from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import *
from tkinter import filedialog
import os

window = Toplevel()
#window = Tk()
window.title('lncLocPred:predict the subcellular localization of lncRNA')
window.geometry('600x400')

#image
canvas = Canvas(window,height=200,width=550)
image_file = PhotoImage(file='1324.png')
image = canvas.create_image(0,0,anchor='nw',image=image_file)
canvas.pack(side='top')


def openfasta():
    messagebox.showinfo(message='Select fasta file')
    filename = filedialog.askopenfilename(title='Select fasta file')
    filename = filename.replace('/','\\')

    father_path = os.path.abspath(os.path.dirname(filename)+os.path.sep+".")
    print(father_path)
    return filename,father_path

def openpse():
    messagebox.showinfo(message='Select PseDNC file')
    psename = filedialog.askopenfilename(title='Select PseDNC file')
    print(psename)
    return psename

def opentri():
    messagebox.showinfo(message='Select Triplet file')
    triname = filedialog.askopenfilename(title='Select Triplet file')  ##,parent=window
    print(triname)
    return triname


def predictloc():
    import numpy as np
    import matlab.engine
    seqfilename,father_path = openfasta()
    psedncfilename = openpse()
    tripletfilename = opentri()
    messagebox.showinfo(title='Please wait',message='The program is predicting, a prompt will pop up when the prediction is over, please wait.')
    eng = matlab.engine.start_matlab()
    kmernor8,kmernorclorder8 = np.asarray(eng.KmernorCL112(seqfilename,8,nargout=2))
    kmerfeature8 = kmernorclorder8[:,0:9080]
    import re
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
    
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.pipeline import  Pipeline
    from sklearn.linear_model import LogisticRegression
    steps = []
    steps.append(('MinMaxScaler', MinMaxScaler(feature_range=(-1, 1))))
    steps.append(('LR', LogisticRegression(C =1000, multi_class='multinomial', solver='lbfgs')))
    model = Pipeline(steps)
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
    
    output = open(father_path+'\\locationResult.txt','w',encoding='gbk')
    output.write('SequenceOrder,\t Location,\t Probability\n')
    for row in resultt:	
        rowtxt = '{},\t {},\t {}'.format(row[0],row[1],row[2])	
        output.write(rowtxt)	
        output.write('\n')
    output.close()
    
    messagebox.showinfo(title='End of prediction',message='The prediction result `` locationResult.txt '' has been saved in the current folder.')

button = Button(window,text = 'Start Prediction',font=('Times New Roman',16, 'bold'),fg='blue',command=predictloc)
button.place(x=217,y=234)

window.mainloop()
