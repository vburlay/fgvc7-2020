import os
import keras._tf_keras.keras as keras
import tensorflow as tf
import numpy as np
from tutorial import dat_gen
import pandas as pd

_, _, generator_train_full = dat_gen.train_val_generators()
classes = dict((v,k) for k,v in generator_train_full.class_indices.items())
#
base_dir = os.getcwd()
tabels_dir = os.path.join(base_dir, 'data/customer_model_resnet.csv')
models_dir =  os.path.join(base_dir, 'models/customer_model_resnet.keras')

test_dir = os.path.join(base_dir, 'data/test')
df_name = pd.DataFrame({'date':[f.replace('.jpg','') for f in os.listdir(test_dir)],
                       'path':[os.path.join(test_dir, f) for f in os.listdir(test_dir)]})
df_sample = pd.read_csv(os.path.join(base_dir, 'data/sample_submission.csv'))
df_sample.drop(columns=['healthy', 'multiple_diseases', 'rust', 'scab'], axis=1, inplace=True)
def predict_col(df_name,col,models_dir):
    model = keras.models.load_model(models_dir)
    ls = []
    for i in df_name.path:
        img = keras.utils.load_img(path=i, target_size=(224, 224), color_mode='rgb')
        img_array = keras.utils.img_to_array(img)
        img_array /= 255  # Scaling
        img_array = tf.expand_dims(img_array, 0)  # Create a batch
        image_tensor = np.hstack([img_array])
        predictions = model.predict(image_tensor)
        score = tf.nn.softmax(predictions[0]).numpy()
        if col == 'healthy':
            ls.append(score[0])
        elif col == 'multiple_diseases':
            ls.append(score[1])
        elif col == 'rust':
            ls.append(score[2])
        elif col == 'scab':
            ls.append(score[3])
    return pd.DataFrame({col:ls})

df_tmp = (df_name
          .assign(healthy=lambda df_: predict_col(df_, 'healthy', models_dir))
          .assign(multiple_diseases=lambda df_: predict_col(df_, 'multiple_diseases', models_dir))
          .assign(rust=lambda df_: predict_col(df_, 'rust', models_dir))
          .assign(scab=lambda df_: predict_col(df_, 'scab', models_dir))
          )
df_test = df_sample.set_index('image_id').join(df_tmp.set_index('date')).drop(columns='path')
df_test.to_csv(tabels_dir)


