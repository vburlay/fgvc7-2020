
            """ Created on Sat Juni 22 15:54:49 2024
                @author: Vladimir Burlay
            """
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

            elif add_radio == "ResNet50(pre-train)":
                models_dir = os.path.join(base_dir, 'models/resnet50.keras')

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
    