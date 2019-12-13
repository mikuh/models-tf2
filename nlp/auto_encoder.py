"""自编码器
"""
import numpy as np
from tensorflow import keras

input_dim = 784
hidden_dim = 64
hidden_dim_m = 32


def load_data():
    x_train = np.random.random([2000, input_dim])
    x_test = np.random.random([100, input_dim])
    print(x_train.shape)
    return x_train, x_test


def auto_encoder_fc():
    model = keras.Sequential([
        keras.layers.Dense(hidden_dim, activation='relu'),
        keras.layers.Dense(hidden_dim_m, activation='relu'),
        keras.layers.Dense(hidden_dim, activation='relu'),
        keras.layers.Dense(input_dim, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model


if __name__ == '__main__':
    x_train, x_test = load_data()
    model = auto_encoder_fc()
    # model.summary()
    model.fit(x_train, x_train, batch_size=50, epochs=50)

    model.summary()