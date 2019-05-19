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
## 二.计算机视觉快速特征描述子--稠密SIFT（dense sift）
dense SIFT在目标分类和场景分类有重要的应用。该算法首先将表达目标的区域分成相同大小的区域块,计算每一个小块的SIFT特征,再对各个小块的稠密SIFT特征在中心位置进行采样,建模目标的表达。然后度量两个图像区域的不相似性,先计算两个区域对应小块的巴氏距离,再对各距离加权求和作为两个区域间的距离。因为目标所在区域靠近边缘的部分可能受到背景像素的影响,而区域的内部则更一致,所以越靠近区域中心权函数的值越大。  
普通的SIFT在之前的博客中有详细的介绍，这里主要讲一下二者的不同点。dense SIFT是提取我们感兴趣的区域中的每个位置的SIFT特征。而通常做特征匹配的SIFT算法只是得到感兴趣区域或者图像上若干个稳定的关键点的SIFT特征。总而言之，当研究目标是对同样的物体或者场景寻找对应关系时，SIFT更好。而研究目标是图像表示或者场景理解时，Dense SIFT更好，因为即使密集采样的区域不能够被准确匹配，这块区域也包含了表达图像内容的信息。  
利用如下的例子，可以计算dense SIFT描述子，并可视化它们的位置：  
~~~python
# -*- coding: utf-8 -*-
from PCV.localdescriptors import sift, dsift
from pylab import  *
from PIL import Image

dsift.process_image_dsift('gesture/empire.jpg','empire.dsift',90,40,True)
l,d = sift.read_features_from_file('empire.dsift')
im = array(Image.open('gesture/empire.jpg'))
sift.plot_features(im,l,True)
title('dense SIFT')
show()
~~~
其他相关代码可以用之前博客介绍过的执行角本通过添加一些额外的参数来得到稠密sift特征。  
使用用于定位描述子的局部梯度方向（force_orientation设置为真），该代码可以在整个图像中计算出dense SIFT特征。下图显示了这些位置：  
![image](2.jpg)  
## 三.图像分类：手势识别
在这个示例中，使用dense SIFT描述子来表示这些手势图像，并建立一个简单的手势识别系统。我这里用自己的手进行比划拍照，共有6种手势，每种40张图片。  
下面这段代码展示了6类简单手势图像的dense SIFT描述子：  
~~~python
# -*- coding: utf-8 -*-
import os
from PCV.localdescriptors import sift, dsift
from pylab import  *
from PIL import Image

imlist=['gesture/image2/feichang01.jpg','gesture/image2/er01.jpg',
        'gesture/image2/san01.jpg','gesture/image2/wu01.jpg',
        'gesture/image2/damu01.jpg','gesture/image2/xiaomu01.jpg']

figure()
for i, im in enumerate(imlist):
    print (im)
    dsift.process_image_dsift(im,im[:-3]+'dsift',10,5,True)
    l,d = sift.read_features_from_file(im[:-3]+'dsift')
    dirpath, filename=os.path.split(im)
    im = array(Image.open(im))
    #显示手势含义title
    titlename=filename[:-14]
    subplot(2,3,i+1)
    sift.plot_features(im,l,True)
    title(titlename)
show()
其中，10，5指的是对图像进行SIFT特征计算的次数。  
~~~
![image](3.jpg)  

下面我们可以使用一些代码来读取训练集与测试集，我采用的数据集为自己拍摄的自己左手照片，其中训练集180张，测试集50-60张。  
需要**特别注意**的是：图片的命名方式决定了最后的混淆矩阵的布局，这里推荐使用**字母+uniform+编号**的方式，如“B-uniform01”。  
~~~python
# -*- coding: utf-8 -*-
from PCV.localdescriptors import dsift
import os
from PCV.localdescriptors import sift
from pylab import *
from PCV.classifiers import knn

def get_imagelist(path):
    """    Returns a list of filenames for
        all jpg images in a directory. """

    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]

def read_gesture_features_labels(path):
    # create list of all files ending in .dsift
    featlist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.dsift')]
    # read the features
    features = []
    for featfile in featlist:
        l,d = sift.read_features_from_file(featfile)
        features.append(d.flatten())
    features = array(features)
    # create labels
    labels = [featfile.split('/')[-1][0] for featfile in featlist]
    return features,array(labels)

def print_confusion(res,labels,classnames):
    n = len(classnames)
    # confusion matrix
    class_ind = dict([(classnames[i],i) for i in range(n)])
    confuse = zeros((n,n))
    for i in range(len(test_labels)):
        confuse[class_ind[res[i]],class_ind[test_labels[i]]] += 1
    print ('Confusion matrix for')
    print (classnames)
    print (confuse)

filelist_train = get_imagelist('gesture/train')
filelist_test = get_imagelist('gesture/test')
imlist=filelist_train+filelist_test

# process images at fixed size (50,50)
for filename in imlist:
    featfile = filename[:-3]+'dsift'
    dsift.process_image_dsift(filename,featfile,10,5,resize=(50,50))

features,labels = read_gesture_features_labels('gesture/train/')
test_features,test_labels = read_gesture_features_labels('gesture/test/')
classnames = unique(labels)

# test kNN
k = 1
knn_classifier = knn.KnnClassifier(labels,features)
res = array([knn_classifier.classify(test_features[i],k) for i in
range(len(test_labels))])
# accuracy
acc = sum(1.0*(res==test_labels)) / len(test_labels)
print ('Accuracy:', acc)

print_confusion(res,test_labels,classnames)

~~~
![image](5.jpg)  
上图展示我个人数据集的一部分。  
![image](4.jpg)  
最终，我这个示例的正确率达到了98%，其混淆矩阵如上图：  
