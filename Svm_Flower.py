# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 14:20:48 2023

@author: zhaodf
"""
#采用svm对iris数据集进行分类
#iris数据集的中文名是安德森鸢尾花卉数据集
#英文全称是Anderson’s Iris data set
#iris包含150个样本，对应数据集的每行数据,每行数据包含每个样本的四个特征
#所以iris数据集是一个150行4列的二维表
#通俗地说，iris数据集是用来给花做分类的数据集
#每个样本包含了花萼长度、花萼宽度、花瓣长度、花瓣宽度四个特征
#我们需要建立一个分类器，分类器可以通过样本的四个特征来判断样本属于山鸢尾、变色鸢尾还是维吉尼亚鸢尾

from sklearn import svm
from  sklearn.model_selection  import  train_test_split
from sklearn.datasets import load_iris
# ************************** 数据加载 **************************
iris = load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test  =  train_test_split(x, y, train_size = 0.7)
# ************************** 模型实现 **************************
# 使用sklearn库中的svm实现分类任务
# clf  =  svm.LinearSVC(max_iter=10000) #线性多维向量机，默认参数
clf  =  svm.SVC(C = 0.8 , kernel = 'rbf' , gamma = 1)   # 非线性多维向量机，选用高斯核函数，超参数gamma设置为1，
                                                        # 惩罚参数C，C越大，对误分类的惩罚越大，分类准确率更高，但泛化能力弱；     
                                                        # C越小，对误分类的惩罚越小，分类准确率变低，但泛化能力强  
    
clf.fit(x_train, y_train)  #训练

                                                   
print('训练精度:', clf.score(x_train, y_train))
print('测试精度:', clf.score(x_test, y_test))
