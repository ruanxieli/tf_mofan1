# -*- coding: utf-8 -*-
# @Time    : 2018/5/25 下午4:10
# @Author  : Xieli Ruan
# @Site    : 
# @File    : tf_mofan1.py
# @Software: PyCharm
import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# create data
x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*0.1+0.3

#create tensorflow structure start

# 随机数列生成了一维，范围-1～1
Weights=tf.Variable(tf.random_uniform([1],-1.0,1.0))
# 初始值为0
biases=tf.Variable(tf.zeros([1]))

y=Weights*x_data+biases

loss=tf.reduce_mean(tf.square(y-y_data))

# 优化器GradientDescentOptimizer：学习效率<1
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss)
init=tf.global_variables_initializer()

#create tensorflow structure end

sess=tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step %20==0:
        print(step,sess.run(Weights),sess.run(biases))