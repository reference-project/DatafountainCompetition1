'''
Created on 2019-3-22
再次测试
@author: czh
'''
import numpy as np
import quanzhong
import tensorflow as tf
import xlrd
from xlutils.copy import copy 
m=100
pathxieru="D:\\data\\AIcompetition\\zhinengpingfen\\test_dataset\\train_dataset.xls"

efile= xlrd.open_workbook(pathxieru)
cfile=copy(efile)
ws=cfile.get_sheet(0)
esheet1=efile.sheet_by_index(0)

for n in range(1,m):    #####A2补0  
    ws.write(n,29,"100")
    print(n)
##cfile.save(pathxieru)