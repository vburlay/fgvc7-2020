from . import dat_gen


def train_vgg16():
    _, _, generator_train_full = dat_gen.train_val_generators()
    pre_trained_model = dat_gen.create_pre_trained_model()
    last_output = dat_gen.output_of_last_layer(pre_trained_model)

    model = dat_gen.get_model_vgg16(pre_trained_model, last_output)
    history = model.fit(generator_train_full,
                        epochs=dat_gen.G.nb_epochs,
                        callbacks=dat_gen.G.callbacks_train)
#     Plot the accuracy and loss
    dat_gen.plot_graphs_train(history, "accuracy")
    dat_gen.plot_graphs_train(history, "loss")

    print(model.evaluate(generator_train_full))
    model.save('vgg16.h5')
#    Y_pred = model.predict(generator_validation)
#    y_pred = np.argmax(Y_pred, axis=1)

def train_eff():
    _, _, generator_train_full = dat_gen.train_val_generators()
    pre_trained_model = dat_gen.create_pre_trained_model_eff()
    last_output = dat_gen.output_of_last_layer_eff(pre_trained_model)

    model = dat_gen.get_model_eff(pre_trained_model, last_output)
    history = model.fit(generator_train_full,
                        epochs=dat_gen.G.nb_epochs,
                        callbacks=dat_gen.G.callbacks_train)
#     Plot the accuracy and loss
    dat_gen.plot_graphs_train(history, "accuracy")
    dat_gen.plot_graphs_train(history, "loss")

    print(model.evaluate(generator_train_full))
    model.save('EfficientNetB0.h5')
#    Y_pred = model.predict(generator_validation)
#    y_pred = np.argmax(Y_pred, axis=1)
def train_customer_model():
    _, _, generator_train_full = dat_gen.train_val_generators()


    model = dat_gen.get_custom_model()
    history = model.fit(generator_train_full,
                        epochs=dat_gen.G.nb_epochs,
                        callbacks=[dat_gen.G.callbacks_train])
#     Plot the accuracy and loss
    dat_gen.plot_graphs_train(history, "accuracy")
    dat_gen.plot_graphs_train(history, "loss")

    print(model.evaluate(generator_train_full))
    model.save('customer_model.keras')

#    Y_pred = model.predict(generator_validation)
#    y_pred = np.argmax(Y_pred, axis=1)


def train_customer_model_vgg():
    _, _, generator_train_full = dat_gen.train_val_generators()


    model = dat_gen.MyVGG(num_classes=4)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(generator_train_full,
                        epochs=dat_gen.G.nb_epochs,
                        callbacks=dat_gen.G.callbacks_train)
#     Plot the accuracy and loss
    dat_gen.plot_graphs_train(history, "accuracy")
    dat_gen.plot_graphs_train(history, "loss")

    print(model.evaluate(generator_train_full))
    model.save('customer_model_vgg.h5')
#    Y_pred = model.predict(generator_validation)
#    y_pred = np.argmax(Y_pred, axis=1)

def train_customer_model_resnet():
    _, _, generator_train_full = dat_gen.train_val_generators()


    model = dat_gen.get_resnet34()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(generator_train_full,
                        epochs=dat_gen.G.nb_epochs,
                        callbacks=dat_gen.G.callbacks_train)
#     Plot the accuracy and loss
    dat_gen.plot_graphs_train(history, "accuracy")
    dat_gen.plot_graphs_train(history, "loss")

    print(model.evaluate(generator_train_full))
    model.save('customer_model_resnet.keras')
#    Y_pred = model.predict(generator_validation)
#    y_pred = np.argmax(Y_pred, axis=1)



