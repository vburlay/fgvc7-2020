# **The Plant Pathalogy Challenge 2020 data set to classify foliar disease of apples** 
![image](https://github.com/vburlay/fgvc7-2020/raw/master/data/fpls-12-723294-g001.PNG) 
A study of the foliar disease of apples for the purpose to:
1) Accurately classify a given image from testing dataset into different diseased category or a healthy leaf;
2) Accurately distinguish between many diseases, sometimes more than one on a single leaf;
3) Deal with rare classes and novel symptoms;
4) Address depth perception—angle, light, shade, physiological age of the leaf;
5) Incorporate expert knowledge in identification, annotation, quantification, and guiding computer vision to search for relevant features during learning. 
> Live demo [_here_](https://appdemopy-fwtelgtpyyazfm3ahlctme.streamlit.app/).
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
- Ubuntu 22.04

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

| Architecture    | Accuracy of Training data |
|-----------|:-------------------------:|
|Customer model  |           0,86            |
|Customer model (ResNet34) |           0,85            |
|ResNet50  |           0,96            |
|VGG16 |           0,83            |
|EfficientNetB0 |           0,96            |


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
The result of Customer model (0.86) - 84 % is good but with pre-train Model (ResNet50 or EfficientNetB0 ) can be better - 96 %:
```r
def feature_extractor(inputs):
    feature_extractor = ResNet50(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')(inputs)
    return feature_extractor
def classifier(inputs):
    x = keras.layers.GlobalAveragePooling2D()(inputs)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(1024, activation="relu")(x)
    x = keras.layers.Dense(512, activation="relu")(x)
    x = keras.layers.Dense(4, activation="softmax", name="classification")(x)
    return x
def final_model(inputs):

    resnet_feature_extractor = feature_extractor(inputs)
    classification_output = classifier(resnet_feature_extractor)

    return classification_output
def get_resnet50():
    inputs = keras.Input(shape=(G.img_height,G.img_width,G.RGB))

    classification_output = final_model(inputs)
    model = keras.Model(inputs=inputs, outputs=classification_output)

    model.compile(optimizer='SGD',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    keras.backend.clear_session()
    return model
```


## Project Status
Project is: _complete_ 


## Room for Improvement

- The data implemented in the analysis has a relatively small volume. This should be improved by the new measurements of the characteristics.
- It is also conceivable that the further number of new classes will be included in the analysis. In this way, the new characteristics of the foliar disease of apples  can make the results more meaningful.



## Contact
Created by [Vladimir Burlay](wladimir.burlay@gmail.com) - feel free to contact me!



