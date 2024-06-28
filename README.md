# **The Plant Pathalogy Challenge 2020 data set to classify foliar disease of apples** 
![image](https://github.com/vburlay/fgvc7-2020/raw/master/data/fpls-12-723294-g001.PNG) 
A study of the foliar disease of apples for the purpose to:
1) Accurately classify a given image from testing dataset into different diseased category or a healthy leaf;
2) Accurately distinguish between many diseases, sometimes more than one on a single leaf;
3) Deal with rare classes and novel symptoms;
4) Address depth perception—angle, light, shade, physiological age of the leaf;
5) Incorporate expert knowledge in identification, annotation, quantification, and guiding computer vision to search for relevant features during learning. 
> Live demo [_here_](https://vburlay-anw-feld-ba-workflowsstreamlit-demo-60fscu.streamlit.app/).
## Table of Contents
* [Genelal Info](#general-nformation)
* [Technologies Used](#technologies-used)
* [Features](#features)
* [Screenshots](#screenshots)
* [Setup](#setup)
* [Usage](#usage)
* [Project Status](#project-status)
* [Room for Improvement](#room-for-improvement)
* [Contact](#contact)


## General Information
> In this project, a major goal was to create a current model. This current model should help with
> diagnosis of the many diseases impacting apples.
> Data set "Plant Pathology" comes from Kaggle [_here_](https://www.kaggle.com/c/plant-pathology-2020-fgvc7).


## Technologies Used
- Python - version 3.10.0

## Features
- Keras(Customer model, Customer model (ResNet34), ResNet50, VGG16, EfficientNetB0)

## Screenshots
* **Accuracy(VGG16,ResNet50)**

VGG16
![image](https://github.com/vburlay/fgvc7-2020/raw/master/my_models_eval/accurancy_vgg16.png) 
ResNet50
![image](https://github.com/vburlay/fgvc7-2020/raw/master/my_models_eval/accurancy_resnet50.png) 

* **Loss (VGG16,ResNet50)** 

VGG16
![image](https://github.com/vburlay/fgvc7-2020/raw/master/my_models_eval/loss_vgg16.png) 

ResNet50
![image](https://github.com/vburlay/fgvc7-2020/raw/master/my_models_eval/loss_resnet50.png) 


* **All models(Evaluation)**

| Architecture    |Accuracy of Training data   |Accuracy of Test data  |
|-----------|:-----:| -----:|
|Customer model  |  0,96     |   0,91    |
|Customer model (ResNet34) |  0,95     |   0,94    |
|ResNet50  |  0,96     |   0,91    |
|VGG16 |  0,99    |   0,94    |
|EfficientNetB0 |  0,94     |   0,94    |


* **Customer model(ResNet34)**

![image3](https://github.com/vburlay/fgvc7-2020/raw/master/model.png) 


## Setup
You can install the package as follows:
```r
ifrom keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from dataclasses import dataclass
from keras._tf_keras.keras.applications.resnet50 import preprocess_input
import os
import tensorflow as tf
import keras._tf_keras.keras
from keras import layers
from keras._tf_keras.keras.layers import Layer
from keras._tf_keras.keras.applications.vgg16 import VGG16
from keras._tf_keras.keras.applications.efficientnet import EfficientNetB0
from keras._tf_keras.keras.applications.resnet50 import ResNet50
import matplotlib.pyplot as plt
```

## Usage
The result 0.94025 - 94 % is good but with preprocessing by clustering the accuracy can be improved. Clustering (K-Means) can be an efficient approach for dimensionality reduction but for this a pipeline has to be created that divides the training data into clusters 34 and replaces the data by their distances to this cluster 34 to apply a logistic regression model afterwards:
```r
pipeline = Pipeline([
    ("kmeans", KMeans(n_clusters = d)),
    ("log_reg", LogisticRegression(multi_class = 'ovr',
             class_weight = None, 
             solver= 'saga', 
             max_iter = 10000)),
])
```


## Project Status
Project is: _complete_ 


## Room for Improvement

- The data implemented in the analysis has a relatively small volume. This should be improved by the new measurements of the characteristics.
- It is also conceivable that the further number of new customer groups will be included in the analysis. In this way, the new characteristics of customers can make the results more meaningful.



## Contact
Created by [Vladimir Burlay](wladimir.burlay@gmail.com) - feel free to contact me!



