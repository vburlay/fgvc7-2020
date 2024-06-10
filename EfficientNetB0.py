
from tutorial import dat_gen, train

if (__name__) == '__main__':

#     generator_train, generator_validation, _ = dat_gen.train_val_generators()
#
#     pre_trained_model = dat_gen.create_pre_trained_model_eff()
#     last_output = dat_gen.output_of_last_layer_eff(pre_trained_model)
#
#     model = dat_gen.get_model_eff(pre_trained_model, last_output)
#     history = model.fit(generator_train,
#                        validation_data=generator_validation,
#                        epochs=dat_gen.G.nb_epochs,
#                        callbacks = dat_gen.G.callbacks )
#     model.summary()
#
#     print(f'This is number ot trainable weightd: {len(model.trainable_weights)}')
#
# #     Plot the accuracy and loss
#     dat_gen.plot_graphs(history, "accuracy")
#     dat_gen.plot_graphs(history, "loss")

#    print(model.evaluate(generator_validation))

    train.train_eff()

