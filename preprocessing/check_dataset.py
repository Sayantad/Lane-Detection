# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 12:22:14 2018

@author: INSAYDAS
"""

import numpy as np
import os
import glob
import scipy.misc as misc
import tensorflow as tf

'''
Checks all images from the  dataset if they have corrupted jpegs, and lists them for removal.

Removal must be done manually from jpeg images and label images!
'''

base_images_path ='../data/binary_lane_bdd/Labels'


image_list =""
final_list = ""

def parse_data(imgfilename):
    image = tf.read_file(imgfilename)
    image = tf.image.decode_jpeg(image, channels = 3)
    return image;

sess = tf.Session()
with sess.as_default():
    sess.run(tf.global_variables_initializer())
    
    count = 0
    fn = tf.placeholder(dtype = tf.string)
    img = parse_data(fn)
   
    for eachfile in os.listdir(base_images_path):
        try:
            base_path = os.path.join(base_images_path,eachfile)
            sess.run(img, feed_dict={fn: base_path})
            image_list += eachfile
        except Exception as e:
            print(base_path, "failed to load !")
            count += 1
            continue
        
    print(count, "images failed to load !")

with open("custom_train.txt", "w", encoding="utf-8") as file:
    file.write(final_list)
print("All done !")
