# ASIN-master
Created by Hang Zhang from Northwestern Polytechnical University.

## Introduction
ASIN is a novel multi-task network based on point cloud data for machining feature recognition. ASIN realizes machining feature segmentation, machining feature identification, and bottom face identification simultaneously. The final machining feature recognition results are obtained by combining the results of the above three tasks. 

In this repository, we release source codes for ASIN. In addition, we can only release a portion of the part point cloud data due to the need for confidentiality.

## Setup
(1)	cuda 11.0.3
(2)	python 3.8.8
(3)	tensorflow 2.4.0
(4)	h5py 2.10.0
(5)	scikit-learn 0.24.1
(6)	numpy 1.19.2
(7)	matplotlib 3.3.4

The code is tested on Intel Core i9-10980XE CPU, 128GB memory, and NVIDIA GeForce RTX 3090 GPU. 

## Train
(1)	Get the ASIN source code by cloning the repository: https://github.com/HARRIXJANG/ASIN-master.git.
(2)	Create the folder named logdir.
(3)	Download related datasets (point cloud data in .h5): https://drive.google.com/drive/folders/1ux1-LsM1O7J3ufHFS5a0BlARX1qIEP1d?usp=sharing. 
(4)	Put the datasets in the folder data.
(5)	Run python train.py to train the network.

## Test 
(1)	Get the ASIN source code by cloning the repository: https://github.com/HARRIXJANG/ASIN-master.git.
(2)	Download related datasets (point cloud data in .h5): https://drive.google.com/drive/folders/1ux1-LsM1O7J3ufHFS5a0BlARX1qIEP1d?usp=sharing. 
(3)	Put the datasets in the folder data.
(4)	Download related pre-trained ASIN model (.h5): https://drive.google.com/drive/folders/1Ha-Q2G3AzqQI4RZEB_18ZAZIYyMx1FPb?usp=sharing. The pre-trained ASIN model was trained on the released dataset.
(5)	Put pre-trained ASIN model in the folder model.
(6)	Run python test.py to test the network.

## Predict
(1)	Get the ASIN source code by cloning the repository: https://github.com/HARRIXJANG/ASIN-master.git.
(2)	Download related pre-trained ASIN model (.h5): https://drive.google.com/drive/folders/1Ha-Q2G3AzqQI4RZEB_18ZAZIYyMx1FPb?usp=sharing. The pre-trained ASIN model was trained on the released dataset.
(3)	Put the pre-trained ASIN model in the folder models.
(4)	Run python predict.py to predict a part. The result is a text file (.txt). In this text file, the first line is "start", the second line is the name of the part, the third line is the serial number of each face in the part (generated by catia), the fourth line is the category (hole, pocket) corresponding to each face, the fifth line is the identification of the bottom surface (0 represents a non-bottom face, 1 represents a bottom face), the sixth line is the clustering results (each set of faces represents a machining feature), and the eighth line is "end".

## Others
(1)	Run draw_results.py to visualize the result.
(2)	Run draw_original_point_clouds.py to visualize the part point cloud in the dataset.
