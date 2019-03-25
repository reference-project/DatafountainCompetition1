'''
Created on 2019-3-21
一共有28个特征，50000条数据
消费者人群画像—信用智能评分
@author: czh
'''
import numpy as np
##import quanzhong
import tensorflow as tf
##np.set_printoptions(threshold = np.inf) 

path="D:\\data\\AIcompetition\\zhinengpingfen\\train_dataset\\train_dataset.csv"
model_path='D:\\TensorflowWorkspace\\study\\AIcompetition\\zhinengpingfen\\model\\model.ckpt'
BATCH_SIZE=8


content=np.loadtxt(path,delimiter=",",dtype=np.str,encoding='UTF-8')
content_shape=content.shape
m=content_shape[0]
##m=50

##print(content_shape)
content_xp=content[1:m,1:29]         ##50000x28
content_label=content[1:m,29:30]     ##50000x1
print(content_xp.shape)
print(content_label.shape)
##print(quanzhong.quanzhongzhi.shape)
content_xp = content_xp.astype('float64')           ##50000x28
meandata=np.mean(content_xp,axis=0)
stddata=np.std(content_xp,axis=0)
content_xp_mean_std=(content_xp-meandata)/stddata
##content_xp_mean=content_xp-np.mean(content_xp,axis=0)
##content_xp_mean_std=content_xp_mean/np.std(content_xp_mean,axis=0)
print("均值",meandata)
print("方差",stddata)

content_label = content_label.astype('float64')
content_label_min=np.min(content_label)
content_label_max=np.max(content_label)
content_label_norm=(content_label-content_label_min)/(content_label_max-content_label_min)
print("最小",content_label_min)
print("最大与最小差值",content_label_max-content_label_min)
##content_label_mean=content_label-np.mean(content_label,axis=0)
##content_label_mean_std=content_label_mean/np.std(content_label_mean,axis=0)
##print("均值",np.mean(content_label,axis=0))
##print("方差",np.std(content_label_mean,axis=0))
##content_label=0.001*content_label                    ##50000x1
##content_xpqz=content_xp*quanzhong.quanzhongzhi      ##50000x28
##print(content_label)

'''
xianshi=content_xpqz[0:2,:]
print("xianshi",xianshi)
'''
X=content_xp_mean_std       ##50000x28
Y_=content_label     ##50000x1
print(content_label_norm)

##Y_=pinjie2
print(X.shape)
print(Y_.shape)
def get_weight(shape, regularizer):
    w = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):  
    b = tf.Variable(tf.constant(0.01, shape=shape)) 
    return b
    
x = tf.placeholder(tf.float32, shape=(None, 28),name='x')
y_ = tf.placeholder(tf.float32, shape=(None, 1),name='y_')

w1 = get_weight([28,64], 0.0000000001)    
b1 = get_bias([64])
y1 = tf.nn.relu(tf.matmul(x, w1)+b1) ##nx14 14x28 


w2 = get_weight([64,128], 0.0000000001)    
b2 = get_bias([128])
y2 = tf.nn.relu(tf.matmul(y1, w2)+b2) ##nx14 14x28 

w2_2 = get_weight([128,256], 0.0000000001)    
b2_2 = get_bias([256])
y2_2 = tf.nn.relu(tf.matmul(y2, w2_2)+b2_2) ##nx14 14x28 

w2_3 = get_weight([256,512], 0.0000000001)    
b2_3 = get_bias([512])
y2_3 = tf.nn.relu(tf.matmul(y2_2, w2_3)+b2_3) ##nx14 14x28 

w2_4 = get_weight([512,1024], 0.0000000001)    
b2_4 = get_bias([1024])
y2_4 = tf.nn.relu(tf.matmul(y2_3, w2_4)+b2_4) ##nx14 14x28 

w2_5 = get_weight([1024,2048], 0.0000000001)    
b2_5 = get_bias([2048])
y2_5 = tf.nn.relu(tf.matmul(y2_4, w2_5)+b2_5) ##nx14 14x28 

w2_6 = get_weight([2048,1024], 0.0000000001)    
b2_6 = get_bias([1024])
##y2_6 = tf.nn.sigmoid(tf.matmul(y2_5, w2_6)+b2_6) ##nx14 14x28 
y2_6 = tf.nn.sigmoid(tf.matmul(y2_5, w2_6)+b2_6)

w3 = get_weight([1024,1], 0.0000000001)
b3 = get_bias([1])
y = tf.matmul(y2_6, w3)+b3 
##y = tf.nn.sigmoid(tf.matmul(y2_4, w3)+b3)



b111 = tf.constant(value=1,dtype=tf.float32)
y_eval = tf.multiply(y,b111,name='y_eval') 

loss_mse = tf.reduce_mean(tf.square(y-y_))
loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))


#定义反向传播方法：不含正则化
train_step =  tf.train.AdamOptimizer(0.0001).minimize(loss_total)
train_step2 = tf.train.AdamOptimizer(0.00001).minimize(loss_total)
train_step3 = tf.train.AdamOptimizer(0.000001).minimize(loss_total)
train_step4 = tf.train.AdamOptimizer(0.0000001).minimize(loss_total)

saver = tf.train.Saver()
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 5000000
    for i in range(STEPS):
        start = (i*BATCH_SIZE) % 300
        end = start + BATCH_SIZE
        if i<(STEPS/4):
            sess.run(train_step, feed_dict={x:X[start:end], y_:Y_[start:end]})
        if i==(STEPS/4):
            print("学习率发生变化")
        if (STEPS/4)<i<(STEPS/2):
            sess.run(train_step2, feed_dict={x:X[start:end], y_:Y_[start:end]})
        if i==(STEPS/2):
            print("学习率发生变化")
        if (STEPS*(3/4))>i>(STEPS/2):
            sess.run(train_step3, feed_dict={x:X[start:end], y_:Y_[start:end]})
        if i==(STEPS*(3/4)):
            print("学习率发生变化")
        if i>STEPS*(3/4):
            sess.run(train_step4, feed_dict={x:X[start:end], y_:Y_[start:end]})
        if i % 2000 == 0:
            loss_mse_v = sess.run(loss_mse, feed_dict={x:X, y_:Y_})
            print("After %d steps, loss is: %f" %(i, loss_mse_v))
    saver.save(sess,model_path)
sess.close()
