import os
import keras._tf_keras.keras as keras
import tensorflow as tf
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px

classes = {0:'healthy',1:'multiple_diseases',2:'rust',3:'scab'}
base_dir = 'https://raw.githubusercontent.com/vburlay/fgvc7-2020/master'


customer_model = os.path.join(base_dir,'data/customer_model.csv')
customer_model_resnet = os.path.join(base_dir,'data/customer_model_resnet.csv')
vgg16 = os.path.join(base_dir,'data/vgg16.csv')
EfficientNetB0 = os.path.join(base_dir,'data/EfficientNetB0.csv')
resnet50 = os.path.join(base_dir,'data/resnet50.csv')

st.title('The Plant Pathalogy Challenge 2020 data set to classify foliar disease of apples')
st.subheader('An App by Vladimir Burlay')
st.write(('This app classificates sample images from the data set showing symptoms of cedar apple rust(A) apple scrab(B), multiple diseases on a singleleaf (C). Images of symptoms were captured using a digital camera and a smartphone in a research orchard from a lange number of apple cultivars at Cornell AgriTech(Geneva, New York, USA) in 2019'))


st.sidebar.title("Control Panel")
#
with st.sidebar:
    add_selectbox = st.selectbox("App-Mode",["Application start"])
    add_radio = st.radio("Choose a model",("Custom model", "Custom Resnet34", "VGG16(pre-train)", "EfficientNetB0(pre-train)", "ResNet50(pre-train)"))
    if add_radio == "Custom model":
        df_test = pd.read_csv(customer_model)
    elif add_radio == "Custom Resnet34":
        df_test = pd.read_csv(customer_model_resnet)
    elif add_radio == "VGG16(pre-train)":
        df_test = pd.read_csv(vgg16)
    elif add_radio == "EfficientNetB0(pre-train)":
        df_test = pd.read_csv(EfficientNetB0)
    elif add_radio == "ResNet50(pre-train)":
        df_test = pd.read_csv(resnet50)

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
               models_dir = 'models/customer_model.keras'
            elif  add_radio == "Custom Resnet34":
                models_dir = 'models/customer_model_resnet.keras'
            elif add_radio == "VGG16(pre-train)":
                models_dir = 'models/vgg16.h5'
            elif add_radio == "EfficientNetB0(pre-train)":
                models_dir = 'models/EfficientNetB0.h5'
            if add_radio != "ResNet50(pre-train)":
                model = keras.models.load_model(models_dir)
            if res_button and add_radio != "ResNet50(pre-train)":
                image_tensor = np.vstack([img_array])
                predictions = model.predict(image_tensor)
                score = tf.nn.softmax(predictions[0])
                st.write("This image most likely belongs to {} with a {:.2f} percent confidence".format(classes.get(np.argmax(score)), 100 * np.max(score)))
                fr = pd.DataFrame(score).rename(classes)
                fr = fr.rename(columns={0: "Score"})
                st.write(fr)
                st.write(":smile:")
    with tab2:
        ds = df_test[['healthy','multiple_diseases','rust','scab']]
        ds.apply(lambda row: np.argmax(row))
        st.bar_chart(data=ds.apply(lambda row: np.argmax(row)),width=250, height=300)
        st.dataframe(df_test, width=1200, height=600)

    with tab3:
        fig = px.scatter(df_test[['healthy','multiple_diseases','rust','scab']], width=1000, height=650)
        st.plotly_chart(fig)
    with tab4:
        if add_radio == "Custom model":
            custom_model_acc =  os.path.join(base_dir, 'my_models_eval/accurancy.png')
            custom_model_los = os.path.join(base_dir, 'my_models_eval/loss.png')
            st.image(custom_model_acc)
            st.image(custom_model_los)
        elif add_radio == "Custom Resnet34":
            custom_resnet34_acc = os.path.join(base_dir, 'my_models_eval/accurancy_custom_resnet.png')
            custom_resnet34_los = os.path.join(base_dir, 'my_models_eval/loss_custom_resnet.png')
            st.image(custom_resnet34_acc)
            st.image(custom_resnet34_los)
        elif add_radio == "VGG16(pre-train)":
            vgg16_model_acc = os.path.join(base_dir, 'my_models_eval/accurancy_vgg16.png')
            vgg16_model_los = os.path.join(base_dir, 'my_models_eval/loss_vgg16.png')
            st.image(vgg16_model_acc)
            st.image(vgg16_model_los)
        elif add_radio == "EfficientNetB0(pre-train)":
            eff_model_acc = os.path.join(base_dir, 'my_models_eval/accurancy_es.png')
            eff_model_los = os.path.join(base_dir, 'my_models_eval/loss_es.png')
            st.image(eff_model_acc)
            st.image(eff_model_los)
        elif add_radio == "ResNet50(pre-train)":
            resnet50_model_acc = os.path.join(base_dir, 'my_models_eval/accurancy_resnet50.png')
            resnet50_model_los = os.path.join(base_dir, 'my_models_eval/loss_resnet50.png')
            st.image(resnet50_model_acc)
            st.image(resnet50_model_los)







