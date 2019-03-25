'''
Created on 2019-3-24

@author: czh
'''
import numpy as np
import quanzhong
import tensorflow as tf
##np.set_printoptions(threshold = np.inf) 

path="D:\\data\\AIcompetition\\zhinengpingfen\\train_dataset\\train_dataset.csv"
model_path='D:\\TensorflowWorkspace\\study\\AIcompetition\\zhinengpingfen\\model\\model.ckpt'
BATCH_SIZE=8


content=np.loadtxt(path,delimiter=",",dtype=np.str,encoding='UTF-8')
content_shape=content.shape
##m=content_shape[0]
m=5000

##print(content_shape)
content_xp=content[1:m,1:29]         ##50000x28
content_label=content[1:m,29:30]     ##50000x1
print(content_xp.shape)
print(content_label.shape)

content_xp = content_xp.astype('float64') 
content_xp_mean=content_xp-np.mean(content_xp,axis=0)
content_xp_mean_std=content_xp_mean/np.std(content_xp_mean,axis=0)
##np.std(X,axis=0)
##print(content_xp_mean_std[10,:])