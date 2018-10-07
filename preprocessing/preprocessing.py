# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 14:25:19 2018

@author: INSAYDAS
"""

import os
import os.path
#import cv2
import numpy as np
#from shutil import copyfile
#import tensorflow as tf
import sys

from PIL import Image
from PIL import ImageFilter
#import glob
def crop_image(img, box):
    return img.crop(box)
def perf_crop_images(filepath):
    

    img = Image.open(filepath)
    img1_crop = [0, 0, 321, 321]
    img2_crop = [319, 0, 640, 321]
    img3_crop = [639, 0, 960, 321]
    img4_crop = [959, 0, 1280, 321]
    
    img5_crop = [0, 282, 321, 603]
    img6_crop = [319, 282, 640,603]
    img7_crop = [639, 282, 960, 603]
    img8_crop = [959, 282, 1280, 603]
    
    img9_crop = [0, 399, 321, 720]
    img10_crop = [319, 399, 640,720]
    img11_crop = [639, 399, 960, 720]
    img12_crop = [959, 399, 1280,720]
    
    cropped_images = []
    crop_values = np.array([img1_crop, img2_crop, img3_crop,img4_crop, img5_crop, img6_crop, img7_crop, img8_crop, img9_crop, img10_crop, img11_crop, img12_crop])
    for box in crop_values:
        cropped_images.append(crop_image(img, box))
        
    return cropped_images
    


def resize(inp_img_dir,op_img_dir, filenames, out_format,n_channels):
    count = 0
    for i in range(len(filenames)):
        count += 1
        #sys.stdout.write('\r>> Processing image %d shard %s' % (
        #    i + 1, filenames[i]))
        image_filename = os.path.join(inp_img_dir,filenames[i] + "." + "jpg")
        sys.stdout.write('\r>> Processing image %d shard %s' % (
            i + 1, image_filename))
        img = Image.open(image_filename)
        if n_channels == 3:
            img = img.filter(ImageFilter.GaussianBlur(radius=0.3))
        #img = img.filter(ImageFilter.SHARPEN)
        img = img.resize((640,360))
        #img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)

        #for k, img in enumerate(result):
        if out_format == "JPEG":
            output_path = os.path.join(op_img_dir,filenames[i] + "." + "jpg")
            output_path_flip = os.path.join(op_img_dir,filenames[i] +
                                            "_f" + "." + "jpg")
            
            if n_channels == 1:
                if img.mode == "RGB":
                    sys.stdout.write('\r>> Converting RGB to JPG image %d shard %s\n' % (
                            i + 1, output_path))
                    img = img.convert("L")
                    #img_flip = img.convert("L")
            img.save(output_path,"JPEG")
            #img_flip.save(output_path_flip,"JPEG")
    
        if out_format == "PNG":
            output_path = os.path.join(op_img_dir,filenames[i] + "." + "png")
            
            if n_channels == 1 :
                if img.mode == "RGB":
                    sys.stdout.write('\r>> Converting RGB to JPG image %d shard %s\n' % (
                            i + 1, output_path))
                    img = img.convert("L")
                    #img_flip = img.convert("L")    
            img.save(output_path,"PNG") 
            #img_flip.save(output_path_flip,"PNG")
        
            #write_jpeg(img, output_path)
            #output_image = tf.image.encode_jpeg(img)
            #tf.write_file(output_path, output_image)
    sys.stdout.write('\r>> No of image processed %d' % count)

def processing(inp_img_dir,op_img_dir, filenames, out_format):
      
    count = 0
    for i in range(len(filenames)):
        count += 1
        #sys.stdout.write('\r>> Processing image %d shard %s' % (
        #    i + 1, filenames[i]))
        image_filename = os.path.join(inp_img_dir,filenames[i] + "." + "jpg")
        sys.stdout.write('\r>> Processing image %d shard %s' % (
            i + 1, image_filename))
        result=perf_crop_images(image_filename)

        for k, img in enumerate(result):
            output_path = os.path.join(op_img_dir,filenames[i] + "_" + str(k) +"." + "jpg")
            img.save(output_path,"JPEG")
            #write_jpeg(img, output_path)
            #output_image = tf.image.encode_jpeg(img)
            #tf.write_file(output_path, output_image)
    sys.stdout.write('\r>> No of image processed %d' % count)
    
if __name__ == "__main__": 
    #ImageSets_path = "E://LaneMarking//models-master//models-master//research//deeplab/datasets//LaneMarking//ImageSet"
    #Imgpath = '/home/ubuntu/work/dataset-labels/images/'
    #labelpath = '/'
    JpegImagePath = 'D:/mycodeFCN/data/lane_data/JPEGImages'
    LabelImagePath = 'D:/mycodeFCN/data/lane_data/Labels'
    cropedJpegImagePath ='D:/mycodeFCN/data/lane_data_640_360_g/training/JPEGImages'
    cropedLabelImagePath='D:/mycodeFCN/data/lane_data_640_360_g/training/Labels'
    count = 0
    
    #trainval_path = os.path.join(ImageSets_path,"trainval.txt")
    #filenames = [x.strip('\n') for x in open(trainval_path, 'r')]
    filenames = []
    for eachfile in os.listdir(JpegImagePath):
        if os.path.exists(os.path.join(LabelImagePath,eachfile)):
            if eachfile.endswith(".jpg"):
                filenames.append(eachfile.split('.')[0])
    #processing(JpegImagePath, cropedJpegImagePath, filenames, "jpg")
    #processing(LabelImagePath,cropedLabelImagePath, filenames, "jpg")

    resize(JpegImagePath, cropedJpegImagePath, filenames, "JPEG", 3)
    resize(LabelImagePath,cropedLabelImagePath, filenames, "PNG", 1)
#   
#            tf
#            
    
    