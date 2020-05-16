from keras.models import Model
from keras.layers import Dense, Flatten, Input, Dropout, Conv1D, GRU
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l1, l2, l1_l2


def build_model_cnn(n_features,
                window,
                nb_filters = 2,
                final_l2 = 5e-5,
                input_l1 = 2e-4,
                nb_output_bins = 4,
                learning_rate = 1e-3,
                sgd_momentum=0.,
                optimizer = 'adam',
                model_type = 'regression'):

    input = Input(shape=(window, n_features), name='input_part')

    out = Conv1D(n_features,1,padding='causal',name='feature_selection_layer',kernel_regularizer=l1(input_l1))(input)

    out = Conv1D(nb_filters, 2, padding='causal',
                                        name='initial_causal_conv')(out)

    out = Conv1D(nb_output_bins, 1, padding='same',
                               kernel_regularizer=l2(final_l2))(out)

    out = Flatten()(out)

    if model_type == "classification":
        out = Dense(1, activation='sigmoid')(out)
    elif model_type == "regression":
        out = Dense(1, activation='linear')(out)

    model = Model(input, out)

    if optimizer == 'sgd':
        opt = SGD(lr=learning_rate, momentum=sgd_momentum)
    elif optimizer == 'rmsprop':
        opt = RMSprop(lr=learning_rate)
    elif optimizer == 'adam':
        opt = Adam(lr=learning_rate)
        print('using Adam')

    if model_type == "classification":
            model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    elif model_type == "regression":
        model.compile(loss='mean_squared_error',
                  optimizer=opt,
                  metrics=['mean_squared_error'])

    return model


def build_model_gru(n_features,
                    window,
                    nb_hidden_1=100,
                    nb_hidden_2=50,
                    emb_dim=20,
                    dropout_1=0.2,
                    dropout_2=0.2,
                    rec_dropout_1=0.2,
                    rec_dropout_2=0.2,
                    model_type='classification'):

    input = Input(shape=(window, n_features),
                        name='input_part')

    out = GRU(input_shape=(window, n_features),
              units=nb_hidden_1,
              dropout=dropout_1,
              recurrent_dropout=rec_dropout_1,
              return_sequences=True)(input)

    out = GRU(units=nb_hidden_2,
              dropout=dropout_2,
              recurrent_dropout=rec_dropout_2,
              return_sequences=False)(out)

    out = Dense(units=emb_dim,
                activation='linear')(out)

    if model_type == "classification":
        out = Dense(1, activation='sigmoid')(out)
    elif model_type == "regression":
        out = Dense(1, activation='relu')(out)

    model = Model(input, out)

    if model_type == "classification":
        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
    elif model_type == "regression":
        model.compile(loss='mean_absolute_error',
                      optimizer='rmsprop',
                      metrics=['mean_absolute_error'])

    return model
