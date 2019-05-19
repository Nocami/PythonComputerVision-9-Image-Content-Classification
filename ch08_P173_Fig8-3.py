# -*- coding: utf-8 -*-
import os
from PCV.localdescriptors import sift, dsift
from pylab import  *
from PIL import Image

imlist=['gesture/image2/B-uniform01.jpg','gesture/image2/F-uniform01.jpg',
        'gesture/image2/G-uniform01.jpg','gesture/image2/L-uniform01.jpg',
        'gesture/image2/O-uniform01.jpg','gesture/image2/V-uniform01.jpg']

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
