import os
import io
import keras._tf_keras.keras as keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tutorial import dat_gen
import streamlit as st
import pandas as pd
import plotly.express as px
# def predict_col(df_name,col,models_dir):
#     model = keras.models.load_model(models_dir)
#     ls = []
#     for i in df_name.path:
#         img = keras.utils.load_img(path=i, target_size=(224, 224), color_mode='rgb')
#         img_array = keras.utils.img_to_array(img)
#         img_array /= 255  # Scaling
#         img_array = tf.expand_dims(img_array, 0)  # Create a batch
#         image_tensor = np.hstack([img_array])
#         predictions = model.predict(image_tensor)
#         score = tf.nn.softmax(predictions[0]).numpy()
#         if col == 'healthy':
#             ls.append(score[0])
#         elif col == 'multiple_diseases':
#             ls.append(score[1])
#         elif col == 'rust':
#             ls.append(score[2])
#         elif col == 'scab':
#             ls.append(score[3])
#     return pd.DataFrame({col:ls})

# df_tmp = (df_name
#           .assign(healthy=lambda df_: predict_col(df_, 'healthy', models_dir))
#           .assign(multiple_diseases=lambda df_: predict_col(df_, 'multiple_diseases', models_dir))
#           .assign(rust=lambda df_: predict_col(df_, 'rust', models_dir))
#           .assign(scab=lambda df_: predict_col(df_, 'scab', models_dir))
#           )
# df_test = df_sample.set_index('image_id').join(df_tmp.set_index('date')).drop(columns='path')

_, _, generator_train_full = dat_gen.train_val_generators()
classes = dict((v,k) for k,v in generator_train_full.class_indices.items())
#
base_dir = os.getcwd()
tabels_dir = os.path.join(base_dir, 'data')

st.title('The Plant Pathalogy Challenge 2020 data set to classify foliar disease of apples')
st.subheader('An App by Vladimir Burlay')
st.write(('This app classificates sample images from the data set showing symptoms of cedar apple rust(A) apple scrab(B), multiple diseases on a singleleaf (C). Images of symptoms were captured using a digital camera and a smartphone in a research orchard from a lange number of apple cultivars at Cornell AgriTech(Geneva, New York, USA) in 2019'))

# test_dir = os.path.join(base_dir, 'data/test')
# df_name = pd.DataFrame({'date':[f.replace('.jpg','') for f in os.listdir(test_dir)],
#                        'path':[os.path.join(test_dir, f) for f in os.listdir(test_dir)]})
#df_sample = pd.read_csv(os.path.join(base_dir, 'data/sample_submission.csv'))
#df_sample.drop(columns=['healthy', 'multiple_diseases', 'rust', 'scab'], axis=1, inplace=True)

st.sidebar.title("Control Panel")
#
with st.sidebar:
    add_selectbox = st.selectbox("App-Mode",["Application start", "Show the source code"])
    add_radio = st.radio("Choose a model",("Custom model", "Custom Resnet34", "VGG16(pre-train)", "EfficientNetB0(pre-train)"))
    if add_radio == "Custom model":
        tabels_dir = os.path.join(base_dir, 'data/customer_model.csv')
        df_test = pd.read_csv(tabels_dir)
    elif add_radio == "Custom Resnet34":
        tabels_dir = os.path.join(base_dir, 'data/customer_model_resnet.csv')
        df_test = pd.read_csv(tabels_dir)

    elif add_radio == "VGG16(pre-train)":
        tabels_dir = os.path.join(base_dir, 'data/vgg16.csv')
        df_test = pd.read_csv(tabels_dir)

    elif add_radio == "EfficientNetB0(pre-train)":
        tabels_dir = os.path.join(base_dir, 'data/EfficientNetB0.csv')
        df_test = pd.read_csv(tabels_dir)

if add_selectbox  == "Application start" :
    st.title("The Plant")

    tab1,tab2,tab3,tab4 = st.tabs(["Image","Result tabular(test)","Result Diagram","Evaluation"])
    with tab1:
        uploaded_file = st.file_uploader("Choose an image...")
        if uploaded_file is not None:
            img_view = keras.utils.load_img(path=uploaded_file, target_size=(400, 500), color_mode='rgb',
                                   interpolation='nearest')
            img = keras.utils.load_img(path=uploaded_file, target_size=(224, 224), color_mode='rgb')
            img_array = keras.utils.img_to_array(img)
            img_array /= 255 # Scaling
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


        st.dataframe(df_test, width=1200, height=600)



    with tab3:

        st.bar_chart(data=df_test.loc[:, ['healthy', 'multiple_diseases','rust','scab']],  width=1000, height=500)
        fig = px.scatter(df_test, width=1000, height=650)
        st.plotly_chart(fig)
    with tab4:

        st.title('3')

# plt.imshow(keras.utils.img_to_array(img)/255)
# plt.show()







