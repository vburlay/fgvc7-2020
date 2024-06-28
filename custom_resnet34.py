from tutorial import dat_gen, train

if (__name__) == '__main__':
    generator_train, generator_validation, _ = dat_gen.train_val_generators()
    model = dat_gen.get_resnet34()
    # history = model.fit(generator_train,
    #                          validation_data=generator_validation,
    #                          epochs=dat_gen.G.nb_epochs,
    #                          callbacks = dat_gen.G.callbacks )
    model.summary()
    # print(f'This is number ot trainable weightd: {len(model.trainable_weights)}')
    #
    # dat_gen.plot_graphs(history, "accuracy")
    # dat_gen.plot_graphs(history, "loss")

    print(model.evaluate(generator_validation))




#    train.train_customer_model_resnet()
