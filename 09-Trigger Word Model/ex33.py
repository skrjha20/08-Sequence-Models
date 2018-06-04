from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam
from td_utils import *
import matplotlib.pyplot as plt
import numpy as np

def model(input_shape):
    X_input = Input(shape=input_shape)
    X = Conv1D(196, kernel_size=15, strides=4)(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Dropout(0.8)(X)

    X = GRU(units=128, return_sequences=True)(X)
    X = Dropout(0.8)(X)
    X = BatchNormalization()(X)

    X = GRU(units=128, return_sequences=True)(X)
    X = Dropout(0.8)(X)
    X = BatchNormalization()(X)
    X = Dropout(0.8)(X)

    X = TimeDistributed(Dense(1, activation="sigmoid"))(X)  # time distributed  (sigmoid)
    model = Model(inputs=X_input, outputs=X)

    return model


def detect_triggerword(filename):
    plt.subplot(2, 1, 1)

    x = graph_spectrogram(filename)
    # the spectogram outputs (freqs, Tx) and we want (Tx, freqs) to input into the model
    x = x.swapaxes(0, 1)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)

    plt.subplot(2, 1, 2)
    plt.plot(predictions[0, :, 0])
    plt.ylabel('probability')
    plt.show()
    return predictions

if __name__ == "__main__":

    X = np.load("./XY_train/X.npy")
    Y = np.load("./XY_train/Y.npy")
    X_dev = np.load("./XY_dev/X_dev.npy")
    Y_dev = np.load("./XY_dev/Y_dev.npy")

    Tx = 5511
    n_freq = 101
    Ty = 1375

    model = model(input_shape=(Tx, n_freq))
    print(model.summary())

    model = load_model('./models/tr_model.h5')
    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
    model.fit(X, Y, batch_size = 5, epochs=1)

    loss, acc = model.evaluate(X_dev, Y_dev)
    print("Dev set accuracy = ", acc)

