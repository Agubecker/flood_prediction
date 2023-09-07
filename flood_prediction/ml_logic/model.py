from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers, metrics
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow import keras
from keras.callbacks import EarlyStopping

def init_model(X_train, y_train):
    """
    Instanciate, compile and and return a model
    """

    # Normalization
    normalizer = Normalization()
    normalizer.adapt(X_train)

    # RNN architecture
    model = models.Sequential()

    ## All the rows will be standardized through the already adapted normalization layer
    model.add(normalizer)
    model.add(layers.LSTM(64,
                          activation='tanh',
                          return_sequences = False,
                          kernel_regularizer=L1L2(l1=0.05, l2=0.05),
                          ))
    ## Predictive Dense Layers
    output_length = y_train.shape[1]
    model.add(layers.Dense(output_length, activation='linear'))

    # Compiler
    adam = optimizers.Adam(learning_rate=0.02)
    model.compile(loss='mse', optimizer=adam, metrics=["mae"])

    return model

def fit_model(model: keras.Model, X_train, y_train, verbose=1) -> tuple[keras.Model, dict]:
    """
    Fit the `model` object
    """

    es = EarlyStopping(monitor = "val_loss",
                      patience = 3,
                      mode = "min",
                      restore_best_weights = True)


    history = model.fit(X_train, y_train,
                        validation_split = 0.3,
                        shuffle = False,
                        batch_size = 32,
                        epochs = 50,
                        callbacks = [es],
                        verbose = verbose)

    return model, history
