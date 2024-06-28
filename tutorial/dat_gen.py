from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
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


base_dir = os.getcwd()
train_dir = os.path.join(base_dir, 'data/train_full/')

class SimpleDense(Layer):
    def __init__(self,units = 128,activation = None):
        super(SimpleDense,self).__init__()
        self.units = units
        # define the activation to get from the built-in activation layers in Keras
        self.activation = keras._tf_keras.keras.activations.get(activation)
    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(name="kernel",
                             initial_value=w_init(shape=(input_shape[-1], self.units),
                                                  dtype='float32'),
                             trainable=True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(name="bias",
                             initial_value=b_init(shape=(self.units,), dtype='float32'),
                             trainable=True)

    def call(self, inputs):
        # pass the computation to the activation layer
        return self.activation(tf.matmul(inputs, self.w) + self.b)

@dataclass
class G:
    img_height = 224
    img_width = 224
    RGB = 3
    nb_epochs = 100
    batch_size = 64
    num_classes = 4
    # callbacks = [keras.callbacks.EarlyStopping(
    #     monitor="val_loss",
    #     patience=5), keras.callbacks.ReduceLROnPlateau(
    #     monitor='val_loss', patience=5, mode='min', factor=0.2, min_lr=1e-7, verbose=1)]
    callbacks_lr = [keras.callbacks.LearningRateScheduler(
        lambda epoch: 1e-6 * 10 ** (epoch / 20))]
    callbacks = [keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', patience=3, mode='min', factor=0.2, min_lr=1e-7, verbose=0)]
    callbacks_train = [keras.callbacks.ReduceLROnPlateau(
        monitor='loss', patience=3, mode='min', factor=0.2, min_lr=1e-7, verbose=0)]

def train_val_generators():
    gen_train = ImageDataGenerator(
        rescale= 1./255.,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest' ,
        validation_split=0.2,
        preprocessing_function=preprocess_input)
    generator_train = gen_train.flow_from_directory(
        directory=train_dir,
        target_size=(G.img_height, G.img_width),
        batch_size= G.batch_size,
        shuffle=True,
        color_mode="rgb",
        class_mode="categorical",
        subset='training',
        seed=21
    )
    generator_validation = gen_train.flow_from_directory(
        directory=train_dir,
        target_size=(G.img_height, G.img_width),
        batch_size=G.batch_size,
        shuffle=True,
        color_mode="rgb",
        class_mode="categorical",
        subset='validation',
        seed=21
    )
    gen_train_full = ImageDataGenerator(
        rescale= 1./255.,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest' ,
        preprocessing_function=preprocess_input)
    generator_train_full = gen_train_full.flow_from_directory(
        directory=train_dir,
        target_size=(G.img_height, G.img_width),
        batch_size=G.batch_size,
        shuffle=True,
        color_mode="rgb",
        class_mode="categorical",
        seed=21
    )
    return  generator_train, generator_validation, generator_train_full

def create_pre_trained_model():
    pre_trained_model = VGG16(
        include_top=False,
        input_shape=(G.img_height, G.img_width, G.RGB),
        weights="imagenet"
    )
    for layer in pre_trained_model.layers:
        layer.trainable = False
    pre_trained_model.summary()
    print(f'This is number ot trainable weightd: {len(pre_trained_model.trainable_weights)}')
    return pre_trained_model

def output_of_last_layer(pre_trained_model):
    last_desired_layer = pre_trained_model.get_layer("block3_conv1")
    print('last layer output shape: ', last_desired_layer.output.shape)
    last_output = last_desired_layer.output
    print('last layer output: ', last_output)
    return last_output

def create_pre_trained_model_eff():
    pre_trained_model = EfficientNetB0(
        include_top=False,
        input_shape=(G.img_height, G.img_width, G.RGB),
        weights="imagenet"
    )
    pre_trained_model.trainable = True
    # for layer in pre_trained_model.layers[:-5]:
    #     layer.trainable = False
    pre_trained_model.summary()
    return pre_trained_model

def output_of_last_layer_eff(pre_trained_model):
    last_desired_layer = pre_trained_model.get_layer("block7a_se_squeeze")
    print('last layer output shape: ', last_desired_layer.output.shape)
    last_output = last_desired_layer.output
    print('last layer output: ', last_output)
    return last_output


def get_custom_model():
    input = keras.Input(shape=(G.img_height,G.img_width,G.RGB))
    x = layers.Conv2D(128, 3, activation="relu")(input)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64,3,activation="relu")(x)
    x = layers.Dropout(0, 2)(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64,3,activation="relu")(x)
    x = layers.Dropout(0, 2)(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)
#    x = SimpleDense(128,activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0, 2)(x)
    output = layers.Dense(4,activation="softmax")(x)
    model = keras.Model(inputs = input, outputs = output)
    model.compile(optimizer=keras.optimizers.Adam(),
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])

    return model
def get_model_vgg16(pre_trained_model , last_output):
    x = layers.Conv2D(128, 3, activation="relu")(last_output)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.Dropout(0, 5)(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.Dropout(0, 5)(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0, 5)(x)
    x = layers.Dense(4,activation="softmax")(x)
    model = keras.Model(inputs=pre_trained_model.input, outputs=x)

    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-4),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    keras.backend.clear_session()
    return model


class Block(keras._tf_keras.keras.Model):
    def __init__(self, filters, kernel_size, repetitions, pool_size=2, strides=2):
        super(Block, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.repetitions = repetitions
        self.strides = strides
        self.pool_size = pool_size
        # Define a conv2D_0, conv2D_1, etc based on the number of repetitions
        for i in range(repetitions):
            # Define a Conv2D layer, specifying filters, kernel_size, activation and padding.
            vars(self)[f'conv2D_{i}'] = layers.Conv2D(self.filters,
                                                      self.kernel_size,
                                                      activation='relu',
                                                      padding='same')
            # Define the max pool layer that will be added after the Conv2D blocks
            self.max_pool = layers.MaxPooling2D(strides=self.strides, pool_size=self.pool_size)
    def call(self, inputs):
        # access the class's conv2D_0 layer
        conv2D_0 = vars(self)['conv2D_0']
        # Connect the conv2D_0 layer to inputs
        x = conv2D_0(inputs)
        # for the remaining conv2D_i layers from 1 to `repetitions` they will be connected to the previous layer
        for i in range(1, self.repetitions):
            # access conv2D_i by formatting the integer `i`. (hint: check how these were saved using `vars()` earlier)
            conv2D_i = vars(self)[(f'conv2D_{i}')]
            # Use the conv2D_i and connect it to the previous layer
            x = conv2D_i(x)
            # Finally, add the max_pool layer
        max_pool = self.max_pool(x)
        return max_pool
class MyVGG(keras._tf_keras.keras.Model):

    def __init__(self, num_classes):
        super(MyVGG, self).__init__()

#         # Creating blocks of VGG with the following
#         # (filters, kernel_size, repetitions) configurations
        self.block_a = Block(64,3,2)
        self.block_b = Block(128,3,2)
        self.block_c = Block(256,3,3)
        self.block_d = Block(512,3,3)
        self.block_e = Block(512,3,3)
#         # Classification head
#         # Define a Flatten layer
        self.flatten = layers.Flatten()
#         # Create a Dense layer with 256 units and ReLU as the activation function
        self.fc = layers.Dense(256,activation='relu')
#         # Finally add the softmax classifier using a Dense layer
        self.classifier = layers.Dense(num_classes,activation='softmax')

    def call(self, inputs):
#       # Chain all the layers one after the other
        x = self.block_a(inputs)
        x = self.block_b(x)
        x = self.block_c(x)
        x = self.block_d(x)
        x = self.block_e(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.classifier(x)
        return x
# ResNet 34
keras.saving.get_custom_objects().clear()
@keras.saving.register_keras_serializable(package="MyLayers")
class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            layers.Conv2D(filters, 3,  padding='same' , strides=1, use_bias=False,activation="relu"),
            layers.Dropout(0, 5),
            keras.layers.BatchNormalization(),
            self.activation,
            layers.Conv2D(filters, 3,  padding='same' ,strides=1, use_bias=False, activation="relu"),
            layers.Dropout(0, 5),
            keras.layers.BatchNormalization()]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                layers.Conv2D(filters, 3, padding='same' , strides=1, use_bias=False,activation="relu"),
                layers.Dropout(0, 5),
                keras.layers.BatchNormalization()]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)
def get_resnet34():
    model = keras.models.Sequential()
    model.add(layers.Conv2D(64, kernel_size=7, strides=2,
                            input_shape=[224, 224, 3]))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'))
    prev_filters = 64
    for filters in [64] * 1 + [128] * 1:
        strides = 1 if filters == prev_filters else 2
        model.add(ResidualUnit(filters, strides=strides))
        prev_filters = filters
    model.add(keras.layers.GlobalAvgPool2D())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(G.num_classes, activation="softmax"))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    keras.backend.clear_session()
    return model

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
def get_model_eff(pre_trained_model , last_output):
    x = layers.Dense(64, activation="relu")(last_output)
    x = layers.Dropout(0, 5)(x)
    x = layers.Dense(4, activation="softmax")(x)
    model = keras.Model(inputs=pre_trained_model.input, outputs=x)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    keras.backend.clear_session()
    return model
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()
def plot_graphs_train(history, string):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()

    # Adjusting the learning rate (optional)
    #     model_tune = get_model()
    #     lr_history = model_tune.fit(generator_train,
    #                         validation_data=generator_validation,
    #                         epochs=G.nb_epochs,
    #                         callbacks=G.callbacks )
    #     # Definethelearningratearray
    #     lrs = 1e-8 * (10 ** (np.arange(100) / 20))
    #     plt.figure(figsize=(10, 6))
    #     plt.grid(True)
    #     plt.semilogx(lrs, lr_history.history["loss"])
    #     plt.tick_params('both', length=10, width=1, which='both')
    #     plt.axis([1e-8, 1e-5, 0, 3])
    #     plt.show()
    #     keras.backend.clear_session()

