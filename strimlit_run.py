import os
import keras._tf_keras.keras as keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tutorial import dat_gen

_, _, generator_train_full = dat_gen.train_val_generators()
classes = dict((v,k) for k,v in generator_train_full.class_indices.items())
#
base_dir = os.getcwd()
models_dir = os.path.join(base_dir, 'models/customer_model_resnet.keras')
test_dir = os.path.join(base_dir, 'data/train_full/healthy')
data_path = [os.path.join(test_dir,f) for f in os.listdir(test_dir)]
#
model = keras.models.load_model(models_dir)
img = keras.utils.load_img(path=data_path[12], target_size=(224,224),color_mode='rgb',interpolation='nearest')
img_array = keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(classes.get(np.argmax(score)), 100 * np.max(score))
)

plt.imshow(keras.utils.img_to_array(img)/255)
plt.show()







