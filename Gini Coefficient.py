# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 17:52:51 2017

@author: Rodrigo
"""
import pandas as pd
import numpy as np


def gini(actual, pred, cmpcol = 0, sortcol = 1):
     assert( len(actual) == len(pred) )
     all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
     all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
     totalLosses = all[:,0].sum()
     giniSum = all[:,0].cumsum().sum() / totalLosses

     giniSum -= (len(actual) + 1) / 2.
     return giniSum / len(actual)
 
    
    
    
    

def gini_normalized(a, p):
     return gini(a, p) / gini(a, a)
 

actual = [1, 0, 0, 1, 0,]
actual = pd.Series(actual)
actual = actual.values.reshape(1,15)

pred = [0.9, 0.25, 0.8, 0.85, 0.33]
pred2 = [1,0,1,1,0]
predictions = pd.Series(predictions)
predictions = predictions.values.reshape(1,15) 
gini_normalized(actual,predictions)