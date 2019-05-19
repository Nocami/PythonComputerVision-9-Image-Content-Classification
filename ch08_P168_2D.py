# -*- coding: utf-8 -*-
from numpy.random import randn
import pickle
from pylab import *

# create sample data of 2D points
n = 250
# two normal distributions
class_1 = 0.7 * randn(n,2)
class_2 = 1.4 * randn(n,2) + array([5,1])
labels = hstack((ones(n),-ones(n)))
# save with Pickle
#with open('points_normal.pkl', 'wb') as f:
with open('points_normal_test.pkl', 'wb') as f:
    pickle.dump(class_1,f)
    pickle.dump(class_2,f)
    pickle.dump(labels,f)
# normal distribution and ring around it
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
