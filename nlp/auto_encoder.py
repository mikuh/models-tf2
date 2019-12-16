"""自编码器
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


class AutoEncoder(object):

    def __init__(self, type='cnn'):
        x_train, x_test = self.load_data()
        if type == 'cnn':
            self.x_train, self.x_test = self.pre_do_cnn(x_train, x_test)

    def load_data(self):
        """加载数据"""
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        return x_train, x_test

    def pre_do_cnn(self, x_train, x_test):
        """CNN自编码器预处理数据
        """
        x_train = tf.expand_dims(x_train.astype('float32'), -1) / 255.0
        x_test = tf.expand_dims(x_test.astype('float32'), -1) / 255.0
        return x_train, x_test

    def pre_do_fc(self, x_train, x_test):
        """全连接自编码器预处理数据
        """
        x_train = x_train.reshape((-1, 28 * 28)) / 255.0
        x_test = x_test.reshape((-1, 28 * 28)) / 255.0
        return x_train, x_test

    def endocer_fc(self, x):
        """全连接编码器"""

    def decoder_fc(self):
        """全连接解码器"""
        decoder_inputs = keras.Input(shape=(32,), name='encoded_img')

    def encoder_cnn(self):
        """CNN编码器"""
        inputs = keras.layers.Input(self.x_train.shape[1:])
        feature = keras.layers.Conv2D(16, 3, activation='relu')(inputs)
        feature = keras.layers.Conv2D(32, 3, activation='relu')(feature)
        encode = keras.layers.MaxPool2D(3)(feature)
        self.encode_shape = encode.shape
        endocer_model = keras.Model(inputs=inputs, outputs=encode, name='encoder')
        endocer_model.summary()
        return endocer_model

    def decoder_cnn(self):
        """CNN解码器"""
        inputs = keras.Input(shape=self.encode_shape[1:], name='encoded_img')
        feature = keras.layers.UpSampling2D(3)(inputs)
        feature = keras.layers.Conv2DTranspose(16, 3, activation='relu')(feature)
        decode = keras.layers.Conv2DTranspose(1, 3, activation='relu')(feature)
        decoder_model = keras.Model(inputs=inputs, outputs=decode, name='decoder')
        decoder_model.summary()
        return decoder_model

    def auto_enccoder_cnn(self):
        """CNN自编码器"""
        auto_encoder_inputs = keras.Input(self.x_train.shape[1:], name='inputs')
        h = self.encoder_cnn()(auto_encoder_inputs)
        decode = self.decoder_cnn()(h)
        autoencoder = keras.Model(inputs=auto_encoder_inputs, outputs=decode, name='autoencoder')
        autoencoder.summary()
        return autoencoder

    def compile_cnn(self, model='auto'):
        if model == 'auto':
            m = self.auto_enccoder_cnn()
            m.compile(optimizer=keras.optimizers.Adam(),
                      loss=keras.losses.BinaryCrossentropy())
            return m

    def view_result(self, x_test, decoded):
        """可视化展示效果"""
        plt.figure(figsize=(10, 4))
        n = 5
        for i in range(n):
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(x_test[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(2, n, n + i + 1)
            plt.imshow(decoded[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()


if __name__ == '__main__':
    # x_train, y_train, x_test, y_test = load_data()
    # model = auto_encoder_fc()
    # # model.summary()
    # model.fit(x_train, x_train, batch_size=64, epochs=100)
    #
    # model.summary()
    #
    # decoded = model.predict(x_test)

    auto_encoder = AutoEncoder()
    model = auto_encoder.compile_cnn()
    model.fit(auto_encoder.x_train, auto_encoder.x_train, batch_size=64, epochs=100)
    decoded = model.predict(auto_encoder.x_test)
    x_train, x_test = auto_encoder.load_data()
    auto_encoder.view_result(x_test, decoded)
