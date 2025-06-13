# import tensorflow as tf
from tensorflow.python.keras import Model
# from tensorflow.python.keras.layers import (
#     Conv2D,
#     Dense,
#     Flatten,
#     MaxPooling2D,
#     Relu,
# )


class NoteClassifier(Model):
    def __init__(self, input_shape, classes=128):
        super(Model, self).__init__()
        # self.conv1 = Conv2D
