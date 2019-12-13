from tensorflow import keras

input_length = 100


def lstm_model():
    model = keras.Sequential([
        keras.layers.Embedding(input_dim=30000, output_dim=100, input_length=input_length),
        keras.layers.LSTM(32, return_sequences=True),
        keras.layers.LSTM(1, activation='sigmoid', return_sequences=False)
    ])
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
    return model


def gru_model():
    model = keras.Sequential([
        keras.layers.Embedding(input_dim=30000, output_dim=32, input_length=input_length),
        keras.layers.LSTM(32, return_sequences=True),
        keras.layers.LSTM(1, activation='sigmoid', return_sequences=False)
    ])
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
    return model


if __name__ == '__main__':
    lstm = lstm_model()
    lstm.summary()

    gru = gru_model()
    gru.summary()