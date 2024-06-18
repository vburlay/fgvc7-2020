import os
import io
import keras._tf_keras.keras as keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tutorial import dat_gen
import streamlit as st
import pandas as pd


_, _, generator_train_full = dat_gen.train_val_generators()
classes = dict((v,k) for k,v in generator_train_full.class_indices.items())
#
base_dir = os.getcwd()
models_dir = os.path.join(base_dir, 'models/customer_model_resnet.keras')
test_dir = os.path.join(base_dir, 'data/train_full/healthy')
data_path = [os.path.join(test_dir,f) for f in os.listdir(test_dir)]


st.title('The Plant Pathalogy Challenge 2020 data set to classify foliar disease of apples')
st.subheader('An App by Vladimir Burlay')
st.write(('This app classificates sample images from the data set showing symptoms of cedar apple rust(A) apple scrab(B), multiple diseases on a singleleaf (C). Images of symptoms were captured using a digital camera and a smartphone in a research orchard from a lange number of apple cultivars at Cornell AgriTech(Geneva, New York, USA) in 2019')) 


st.sidebar.title("Control Panel")
#
with st.sidebar:
    add_selectbox = st.selectbox("App-Mode",["Application start", "Show the source code"])
    add_radio = st.radio("Choose a model",("Custom model", "Custom Resnet34", "VGG16(pre-train)", "EfficientNetB0(pre-train)"))

if add_selectbox  == "Application start" :
    st.title("The Plant")

    tab1,tab2,tab3 = st.tabs(["Image","Result tabular","Evaluation"])
    with tab1:
        uploaded_file = st.file_uploader("Choose an image...")
        if uploaded_file is not None:
            img_view = keras.utils.load_img(path=uploaded_file, target_size=(400, 500), color_mode='rgb',
                                   interpolation='nearest')
            img = keras.utils.load_img(path=uploaded_file, target_size=(224, 224), color_mode='rgb',
                                            interpolation='nearest')
            img_array = keras.utils.img_to_array(img)
            img_array /= 255
            img_array = tf.expand_dims(img_array, 0)  # Create a batch

            st.image(img_view)
            res_button = st.button("Predict", type="primary")

            if add_radio == "Custom model":
                models_dir = os.path.join(base_dir, 'models/customer_model.keras')

            elif  add_radio == "Custom Resnet34":
                models_dir = os.path.join(base_dir, 'models/customer_model_resnet.keras')

            elif add_radio == "VGG16(pre-train)":
                models_dir = os.path.join(base_dir, 'models/vgg16.h5')

            elif add_radio == "EfficientNetB0(pre-train)":
                models_dir = os.path.join(base_dir, 'models/EfficientNetB0.h5')

            model = keras.models.load_model(models_dir)
            if res_button:
                image_tensor = np.vstack([img_array])
                predictions = model.predict(image_tensor)
                score = tf.nn.softmax(predictions[0])
                st.write("This image most likely belongs to {} with a {:.2f} percent confidence".format(classes.get(np.argmax(score)), 100 * np.max(score)))
                fr = pd.DataFrame(score).rename(classes)
                fr = fr.rename(columns={0: "Score"})
                st.write(fr)
                st.write(":smile:")
    with tab2:
        st.title('2')
    with tab3:
        st.title('3')
# _, _, generator_train_full = dat_gen.train_val_generators()
# classes = dict((v,k) for k,v in generator_train_full.class_indices.items())
# #
# base_dir = os.getcwd()
# models_dir = os.path.join(base_dir, 'models/customer_model_resnet.keras')
# test_dir = os.path.join(base_dir, 'data/train_full/healthy')
# data_path = [os.path.join(test_dir,f) for f in os.listdir(test_dir)]
# #
# model = keras.models.load_model(models_dir)
# img = keras.utils.load_img(path=data_path[12], target_size=(224,224),color_mode='rgb',interpolation='nearest')
# img_array = keras.utils.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0) # Create a batch
#
# predictions = model.predict(img_array)
# score = tf.nn.softmax(predictions[0])
# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(classes.get(np.argmax(score)), 100 * np.max(score))
# )
#
# plt.imshow(keras.utils.img_to_array(img)/255)
# plt.show()







