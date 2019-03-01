# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 19:45:05 2019

@author: stoer
"""

#val_gen
a = val_gen.__dict__
trueList = []
falseList = []
truefalseindex = val_gen.classes
for x in range(len(val_gen.filenames)):
    if truefalseindex[x]==0:
        falseList.append(val_gen.filenames[x])
    else:
        trueList.append(val_gen.filenames[x])

file = open('val_gen_true.txt','w')
file.write(str(trueList))
file.close()