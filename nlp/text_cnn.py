"""
text cnn的简单实现

"""
from tensorflow import keras

num_features = 3000
sequence_length = 300
embedding_dimension = 100

filter_sizes = [3, 4, 5]


def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=num_features)
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=sequence_length)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=sequence_length)
    print(x_train.shape, y_train.shape)
    return x_train, y_train, x_test, y_test


def convolution():
    inn = keras.layers.Input(shape=(sequence_length, embedding_dimension))
    cnns = []
    for size in filter_sizes:
        conv = keras.layers.Conv1D(filters=64, kernel_size=size, activation='relu')(inn)
        pool = keras.layers.MaxPool1D(sequence_length - size + 1)(conv)
        cnns.append(pool)

    outt = keras.layers.concatenate(cnns)
    model = keras.Model(inputs=inn, outputs=outt, name="text_cnn")
    return model


def cnn_mulfilter():
    model = keras.Sequential([
        keras.layers.Embedding(input_dim=num_features, output_dim=embedding_dimension, input_length=sequence_length),
        convolution(),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    model = cnn_mulfilter()
    model.summary()
    keras.utils.plot_model(model, 'textcnn_model.png')
    keras.utils.plot_model(model, 'model_info.png', show_shapes=True)

    x_train, y_train, x_test, y_test = load_data()

    model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.1)

    print(model.evaluate(x_test, y_test, batch_size=50))

    print(model.predict(x_test[:3]))

    model.save('cnn_text_model.h5')
    # 加载模型
    # model = keras.models.load_model('cnn_text_model.h5')
