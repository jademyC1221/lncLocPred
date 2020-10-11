# lncLocPred
The function of long non-coding RNAs is closely related to their subcellular localization. We proposed a machine learning method to identify the subcellular localization of long non-coding RNAs.  

## What does lncLocPred do?
We trained a model that predicts the subcellular localization of the long non-coding RNAs. Given the sequence file (.fasta, .txt, .xlsx) of the long non-coding RNAs, the subcellular localization of the long non-coding RNA can be predicted using our method. This method can predict four subcellular localizations, which are nucleus, cytoplasm, ribosome, and exosome.

## Dependency
Matalb R2018a  
Python3.6  
scikit-learn 0.19.1

## Content
./whole./src:  
The folder named "train" is the training part of the model.  
The folder named "predict" allows people to predict long-chain non-coding subcellular localizations based on our model.  
./whole./supplementary material:  
Contains the benchmark dataset and Independent dataset. 
Contains the The top 1000 features in the optimal feature set.pdf demonstrating our features selected. 
Benchmark dataset is from paper iLoc-lncRNA. 
Independent dataset is dowmloaded from RNALocate. 

## Train
1.Kmercount.m : calculate the k-mer features of the sequences.  
2.KMER_VarianceThreshold.py: we remove those k-mer features with a variance of 0.  
3.Bino.m: sort the k-mer features with a binomial distribution.  
4.IFS.py: Selecte the optimal dimension of k-mer features sorted by binomial distribution.
5.The triplet and SC-PseDNC feature are obtained from the website Pse-in-One 2.0. We used "parameter optimization for pseDNC feature.py" to adjust the parameters.  
6.Fscore.m, write4libsvm.m and fselect.py: Combine the triplet and pseDNC features with the kmer features selected by ifs, and then use w.m to save the feature file in the format of [label 1: value1 2: value2 3: value3 ...]. Run cmd in the path of the formatted file, and enter "python fselect.py filename.txt" to get the Fscore feature ranking result. feature.py is downloaded from https://www.csie.ntu.edu.tw/~cjlin/.
7.IFSofCombinationtp8.py: IFS selection process for combined features. tp8fscorder.mat is the result of the combination of "tp8".  
8.The folder named "traindata"  contains the the selected feature files including different feature combinations and the calculated results for independent dataset.

## Predict
1.You should first store your sequences data for prediction in data.fasta or data.txt or data.xlsx.  
2.You should calculate SC-PseDNC feature and Triplet feature though Pse-in-One 2.0, and datapse.txt and datatriplet.txt are the results you obtain.  
3.Open the lncLocPredTool.py and press F5 to execute. Follow the prompts on the interface to complete the prediction. After the prediction is completed, the results will be saved as a txt file.
4.To increase the speed of prediction, you can use KmernorCL113.m in matlab to get kmernorcl9080.mat, then, put kmernorcl9080.mat together with SC-PseDNC feature and Triplet feature into predictloc.py in python for predicting.

## EXE
Users can download the binary tool from Baidu Netdisk, here is the information:
link：https://pan.baidu.com/s/1LY3CM9BMlRMWjPs42nR-pA 
password：b3ng 
