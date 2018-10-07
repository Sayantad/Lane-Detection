# Lane-Detection
Automatic detection of lanes can be solved by the deep learning based semantic segmentation .Semantic segmentation achieves fine-graned inference by labels for every pixel, so that each pixel is labeled with the class of its enclosing object or region. So, the lane marks in an image can be label with a specific pixel colors which is assigned to the class of the lane marks. The net used is based on fully convolutional neural net described in the paper [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/pdf/1605.06211.pdf). The code is based on implementation of [FCN Implementation] (https://github.com/sagieppel/Fully-convolutional-neural-network-FCN-for-semantic-segmentation-Tensorflow-implementation).


## Requirements
This network was run with Python 3.6  Anaconda package and Tensorflow > 1.1. The training was done using 8 GB Nvidia Quadro P4000, on Windows 10.

## Training
#### Data Preparation
Use the preprocessing/check_dataset.py for checking the jpeg images and labels and manually remove the corroupt image from the data set.

For further preprocessing each images are check manually for improper labelling and removed are from the dataset.

Use the preprocessing/preprocesing.py for downsampling of images to 640 x360. Before down sampling it will also add gaussion blur to remove noise to imput images.This will convert label images to 8bit png images with same name as corresponding images.

The label images use 255 to represent the lane field and 0 for the background.

Finaly divied the data set into training,validation and test set. 10% of dataset is used for validation set, 5% for test set and rest for trainging.

#### Training Model
For training use the TRAIN.py
1) Set folder path of the taining images in TRAIN_Image_Dir
2) Set folder path for the groud truth labels in TRAIN_Label_Dir
3)Download a pretrained [vgg16] (ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy) model and put in the Model folder
4) Set the number of class in NUM_CLASS which is 2 in this case
5) Also set the batchsize to 4.

For Inference use the Inference.py
1) Make sure that train model is logs_dir
2) Set the Image_Dir to the test image folder
3) Set the number of class in NUM_CLASSES

#### Predicted Result
The predicte result after 20000 iterations
![](Result.jpg)

## Future Work
 Future work is involved in increaseing the iou above 0.5.
