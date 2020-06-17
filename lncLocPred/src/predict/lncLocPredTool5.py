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
import tkinter as tk


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

def interface():
    # window = Toplevel()
    window = Tk()
    window.title('lncLocPred')
    window.geometry('600x400')
    
    #image
    canvas = Canvas(window,height=200,width=550)
    image_file = PhotoImage(file='9411.PNG')
    image = canvas.create_image(90,5,anchor='nw',image=image_file)
    canvas.pack(side='top')
    tk.Label(window, text='Predicting LncRNA Subcellular Localization Using Multiple Sequence Feature Information', font=('Times New Roman',12)).place(x=20, y=100, anchor='w')

    button = Button(window,text = 'Input Files for Prediction',font=('Times New Roman',12, 'bold'),fg='white',bg = 'dodgerblue',command=predictloc)
    button.place(x=217,y=170)
    
    howtouse='''
How to use: 
1)  Press "Input Files for Prediction" to execute. Follow the prompts to select prepared datafiles 
     which are stored at the one folder for the prediction.  
2)  Do not close but can minimize the window until the program prompts that prediction is over.
3)  When finished, the results will be saved as `` locationResult.txt ''.
    '''
    
    # tk.Label(window, text=howtouse, font=('Times New Roman',10)).place(x=50, y=234, anchor='w')
    # tk.Label(window, text=howtouse, font=('Times New Roman',10)).place(x=20, y=284, anchor='w')
    tk.Label(window, text=howtouse, font=('Times New Roman',10),justify='left').place(x=25, y=274)  #,fg='slategray'
    
    window.mainloop()

if __name__ == "__main__":
    interface()