'''
Created on 2019-3-22
输入数据进行测试
均值 [618.05306]
方差 [42.442598]
@author: czh
'''
import numpy as np
import quanzhong
import tensorflow as tf
import xlrd
from xlutils.copy import copy  
##np.set_printoptions(threshold = np.inf) 

para1= 297.0
para2=422.0

pathtest="D:\\data\\AIcompetition\\zhinengpingfen\\train_dataset\\train_dataset.csv"
pathxieru="D:\\data\\AIcompetition\\zhinengpingfen\\submit_result.xls"

content=np.loadtxt(pathtest,delimiter=",",dtype=np.str,encoding='UTF-8')
content_shape=content.shape
m=content_shape[0]
##m=5  #测试的样本数量

##print(content_shape)
content_xp=content[1:m,1:29]         ##50000x28
content_label=content[1:m,29:30]     ##50000x1
content_name=content[1:m,0:1]  
print(content_xp.shape)
print(content_label.shape)
print(quanzhong.quanzhongzhi.shape)

content_xp = content_xp.astype('float64')           ##50000x28
content_label = content_label.astype('float64')

##content_xpqz=content_xp*quanzhong.quanzhongzhi      ##50000x28
content_xp_mean_std=content_xp_mean_std=(content_xp-quanzhong.meandata)/quanzhong.stddata
##print(content_xp_mean_std)



data=content_xp_mean_std       ##mx28
print(data)  

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('D:/TensorflowWorkspace/study/AIcompetition/zhinengpingfen/model/model.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('D:/TensorflowWorkspace/study/AIcompetition/zhinengpingfen/model/'))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    feed_dict = {x:data}
    y_eval = graph.get_tensor_by_name("y_eval:0")
    shuchu=sess.run(y_eval,feed_dict)
    ##shuchu=shuchu*para2+para1
    shuchu=shuchu.astype(np.int32)
    ##print(shuchu)
##print(shuchu.shape)


##print(content_name)
for n in range(1,m):     
    print(shuchu[n-1,0])
    
efile= xlrd.open_workbook(pathxieru)
cfile=copy(efile)
ws=cfile.get_sheet(0)
esheet1=efile.sheet_by_index(0)

for n in range(1,m):    #####A2补0  
    ws.write(n,1,str(shuchu[n-1,0]))
    ##print(n)
cfile.save(pathxieru)