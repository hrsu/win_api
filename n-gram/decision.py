import numpy as np
import sklearn.model_selection as train
from sklearn import tree
from sklearn.metrics import accuracy_score
import os

def loadData(filename,type):
    data = np.loadtxt(filename, dtype=type, delimiter=',',skiprows=2)
    x,y=np.split(data,indices_or_sections=(1,),axis=1)
    #后十个为属性值，第一个为标签
    x ,y= y[:,1:],x
    #前十个为属性值
    x_train,x_test,y_train,y_test=train.train_test_split(x,y,random_state=42,train_size=0.8)
    #随机划分训练集与测试集
    return x_train,x_test,y_train,y_test

def Train(x_train,y_train):
    clf = tree.DecisionTreeClassifier()  # 创建DecisionTreeClassifier()类
    clf.fit(x_train, y_train.ravel())
    return clf

def Test(x_train,x_test,y_train,y_test,clf):
    if clf is None:
        raise IOError("Must input a clf!")
    y_hat = clf.predict(x_train)
    score = accuracy_score(y_hat, y_train)
    print('训练集准确率：{}'.format(score))
    y_hat=clf.predict(x_test)
    score=accuracy_score(y_hat,y_test)
    print('测试集准确率：{}'.format(score))


if __name__ == '__main__':
    x_train1, x_test1, y_train1, y_test1 = loadData('sequence.txt', int)
    clf1 = Train(x_train1, y_train1)
    Test(x_train1, x_test1, y_train1, y_test1, clf1)

