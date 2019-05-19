# PythonComputerVision-9-Image-Content-Classification
图像内容分类--本文主要阐述：①knn可视化。②dense sift（稠密sift）原理。③手势识别  
## 一.K邻近分类法(KNN)
目前存在很多分类方法，其中最简单且用的最多的一种方法就是KNN（K-Nearest Neighbor,K邻近分类法），这种算法把要分类的对象，比如我们后面要用到的特征向量，与训练集中已知类标记的所有对象进行对比，并由k近邻对指派到哪个类进行投票。这种方法通常分类效果较好，但是也有很多弊端：与K-means聚类算法一样，需要预先设定K值，K值的选择会影响分类的性能；此外，这种方法要求将整个训练集存储起来，如果训练集非常大，搜索起来会比歼缓慢。对于大训练集，采取某些装箱形式通常会减少对比的次数。  
实现最基本的KNN很简单。这里会展示一个简单的二维示例：  
### 1.创建数据集
源码如下：  

~~~python
# -*- coding: utf-8 -*-
from numpy.random import randn
import pickle
from pylab import *

# create sample data of 2D points
n = 250
# 2个正态分布数据集
class_1 = 0.7 * randn(n,2)
class_2 = 1.4 * randn(n,2) + array([5,1])
labels = hstack((ones(n),-ones(n)))
# save with Pickle
#with open('points_normal.pkl', 'wb') as f:
with open('points_normal_test.pkl', 'wb') as f:
    pickle.dump(class_1,f)
    pickle.dump(class_2,f)
    pickle.dump(labels,f)
# 正态分布并且使数据环绕装分布
print ("save OK!")
class_1 = 0.7 * randn(n,2)
r = 0.9 * randn(n,1) + 5
angle = 3*pi * randn(n,1)
class_2 = hstack((r*cos(angle),r*sin(angle)))
labels = hstack((ones(n),-ones(n)))
# save with Pickle
#with open('points_ring.pkl', 'wb') as f:
with open('points_ring_test.pkl', 'wb') as f:
    pickle.dump(class_1,f)
    pickle.dump(class_2,f)
    pickle.dump(labels,f)
    
print ("save OK!")

~~~  
用不同的保存文件名运行上述代码两次，就可以得到4个二维数据集。每个分布有两个文件，一个训练，一个测试。
### 2.KNN分类器及结果可视化
源码如下：  
~~~python
# -*- coding: utf-8 -*-
import pickle
from pylab import *
from PCV.classifiers import knn
from PCV.tools import imtools

pklist=['points_normal.pkl','points_ring.pkl']

figure()

# load 2D points using Pickle
for i, pklfile in enumerate(pklist):
    with open(pklfile, 'rb') as f:
        class_1 = pickle.load(f)
        class_2 = pickle.load(f)
        labels = pickle.load(f)
    # load test data using Pickle
    with open(pklfile[:-4]+'_test.pkl', 'rb') as f:
        class_1 = pickle.load(f)
        class_2 = pickle.load(f)
        labels = pickle.load(f)

    model = knn.KnnClassifier(labels,vstack((class_1,class_2)))
    # test on the first point
    print (model.classify(class_1[0]))

    #define function for plotting
    def classify(x,y,model=model):
        return array([model.classify([xx,yy]) for (xx,yy) in zip(x,y)])

    # lot the classification boundary
    subplot(1,2,i+1)
    imtools.plot_2D_boundary([-6,6,-6,6],[class_1,class_2],classify,[1,-1])
    titlename=pklfile[:-4]
    title(titlename)
show()
~~~  
上述代码载入训练数据来创建一个KNN分类器模型，并载入另外的测试数据集来对测试数据点进行分类。最后的部分可视化这些测试点（如下图：）  
![image](1.jpg)  
用K邻近分类器分类二维数据，每个示例中，不同颜色代表类标记，正确分类的点用星星表示，错误的用圆点表示，曲线是分类器的决策边界。  
