# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 13:46:48 2019

@author: chenmeijun
"""

from sklearn.feature_selection import *
import numpy as np
from scipy.io import loadmat
import numpy as np
import matlab.engine
eng = matlab.engine.start_matlab()
lncRNA5mer655,lnc5mer655nor = np.asarray(eng.Kmercount(5,nargout=2))
lncRNA6mer655,lnc6mer655nor = np.asarray(eng.Kmercount(6,nargout=2))
lncRNA8mer655,lnc8mer655nor = np.asarray(eng.Kmercount(8,nargout=2))


threshold = 0
def KMER_VarianceThreshold(X,threshold):
    selector = VarianceThreshold(threshold)
    selector.fit(X)
    selector.get_support(indices=False)
    print("Variances is %s" % selector.variances_)
    print("After transform is %s" % selector.transform(X))
    print("The surport is %s" % selector.get_support(True))
    print("After reverse transform is %s" %selector.inverse_transform(selector.transform(X)))
    return selector.get_support(indices=False),selector.variances_,selector.transform(X)

pentamerindex,variances5,lncRNA5mer655 = KMER_VarianceThreshold(X=lncRNA5mer655,threshold=threshold)
hexamerindex,variances6,lncRNA6mer655 = KMER_VarianceThreshold(X=lncRNA6mer655,threshold=threshold)
octamerindex,variances8,lncRNA8mer655_2 = KMER_VarianceThreshold(X=lncRNA8mer655,threshold=threshold)