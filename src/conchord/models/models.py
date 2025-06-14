import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
    ReLU,
)


class NoteClassifier(Model):
    def __init__(self, input_shape=(22, 128, 1), num_classes=128):
        super(NoteClassifier, self).__init__()
        self.num_classes = num_classes
        self.input = Input(shape=input_shape)

        self.Z1 = Conv2D(8, (4, 4), strides=1, padding='same')
        self.A1 = ReLU()
        self.P1 = MaxPooling2D((8, 8), strides=8, padding='same')

        self.Z2 = Conv2D(16, (2, 2), strides=1, padding='same')
        self.A2 = ReLU()
        self.P2 = MaxPooling2D((4, 4), strides=4, padding='same')

        self.F = Flatten()
        self.dropout = Dropout(0.3)
        self.output = Dense(units=num_classes, activation='sigmoid')

    @tf.function
    def call(self, inputs, training=False):
        if tf.rank(inputs) == 3:
            inputs = tf.expand_dims(inputs, axis=-1)
        x = self.Z1(inputs)
        x = self.A1(x)
        x = self.P1(x)
        x = self.Z2(x)
        x = self.A2(x)
        x = self.P2(x)
        x = self.F(x)
        if training:
            x = self.dropout(x)
        return self.output(x)

    # @staticmethod
    # def load_data()
