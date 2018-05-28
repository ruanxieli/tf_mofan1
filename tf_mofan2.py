# -*- coding: utf-8 -*-
# @Time    : 2018/5/26 上午10:36
# @Author  : Xieli Ruan
# @Site    : 
# @File    : tf_mofan2.py
# @Software: PyCharm

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# matrix1=tf.constant([[3,3]])#一行两列
# matrix2=tf.constant(#两行一列
#     [
#         [2],
#         [2]
#     ]
# )
#
# product=tf.matmul(matrix1,matrix2)

# method1
# sess=tf.Session()
# result=sess.run(product)
# print(result)
# sess.close()

# method2: session 自动关闭
# with tf.Session() as sess:
#     result2=sess.run(product)
#     print(result2)

# Variable
# state=tf.Variable(0,name='counter')
# print(state.name)
# one=tf.constant(1)
#
# new_value=tf.add(state,one)
# update=tf.assign(state,new_value)
#
# init=tf.initialize_all_variables()
#
# with tf.Session() as sess:
#     sess.run(init)
#     for _ in range(3):
#         sess.run(update)
#         print(sess.run(state))

# placehoder & feed_dict绑定
# input1=tf.placeholder(tf.float32)
# input2=tf.placeholder(tf.float32)
#
# output=tf.multiply(input1,input2)
# with tf.Session() as sess:
#     print(sess.run(output,feed_dict={input1:[7.],input2:[2.0]}))

def add_layer(inputs, in_size, out_size,activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b=tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs

# -1~1, 300个单位,1个特征
x_data=np.linspace(-1,1,300)[:,np.newaxis]

# 噪点： mean=0，方差0.05, x_data格式
noise=np.random.normal(0,0.05,x_data.shape)

y_data=np.square(x_data)-0.5+noise

xs=tf.placeholder(tf.float32,[None,1])
ys=tf.placeholder(tf.float32,[None,1])

#hiden layer1
l1=add_layer(xs,1,10,activation_function=tf.nn.relu)

# output layer
prediction=add_layer(l1,10,1,activation_function=None)

# reduction_indices=[1]：按行求和
loss= tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))

train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init=tf.global_variables_initializer()

sess=tf.Session()
sess.run(init)

# 图片框
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()#显示后整个程序不暂停
# plt.show()# 显示后整个程序暂停了

for i in range(1000):
    sess.run(train_step, feed_dict={xs:x_data,ys:y_data})
    if i%50:
        # print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))

        try:
            # 去除lines第一个单位
            ax.lines.remove(lines[0])
        except Exception:
            pass

        prediction_value=sess.run(prediction,feed_dict={xs:x_data})
        # 曲线的形式，红色的线，宽度为5
        lines=ax.plot(x_data,prediction_value,'r-',lw=5)
        plt.pause(0.1)




