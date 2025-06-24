from pathlib import Path

import numpy as np
from keras import Model
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, ReLU, TimeDistributed
from tensorflow import expand_dims, rank


class NoteClassifier(Model):
    def __init__(self, input_shape=(22, 128, 1), num_classes=128):
        super(NoteClassifier, self).__init__()
        self.input_shape_ = input_shape

        self.Z1 = Conv2D(8, (4, 4), strides=1, padding='same')
        self.A1 = ReLU()
        # keep time axis (dim 1) for preserving temporal info. pooling and stride: (1,4)
        self.P1 = MaxPooling2D((1, 4), strides=(1, 4), padding='same')

        self.Z2 = Conv2D(16, (2, 2), strides=1, padding='same')
        self.A2 = ReLU()
        self.P2 = MaxPooling2D((1, 4), strides=(1, 4), padding='same')

        self.F = TimeDistributed(Flatten())
        self.dropout = Dropout(0.3)
        self.output_layer = TimeDistributed(Dense(num_classes, activation='sigmoid'))

    def call(self, inputs, training=False):
        if rank(inputs) == 3:
            inputs = expand_dims(inputs, axis=-1)
        x = self.Z1(inputs)
        x = self.A1(x)
        x = self.P1(x)
        x = self.Z2(x)
        x = self.A2(x)
        x = self.P2(x)
        x = self.F(x)
        if training:
            x = self.dropout(x)
        return self.output_layer(x)

    @staticmethod
    def load_data() -> dict[str, np.ndarray]:
        DATA_DIR = Path(__file__).resolve().parents[3] / 'data' / 'splitted'
        model_type = 'notes'

        def load_split(split: str):
            data = np.load(str(DATA_DIR / f'{model_type}_{split}.npz'))
            X: np.ndarray = data['X']
            Y: np.ndarray = data['Y']
            if X.ndim == 3:
                X = np.expand_dims(X, axis=-1)
            return X.astype(np.float32), Y.astype(np.float32)

        X_train, Y_train = load_split('train')
        X_val, Y_val = load_split('val')
        X_test, Y_test = load_split('test')

        return {
            'X_train': X_train,
            'Y_train': Y_train,
            'X_val': X_val,
            'Y_val': Y_val,
            'X_test': X_test,
            'Y_test': Y_test,
        }


if __name__ == '__main__':
    data = NoteClassifier.load_data()
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_test = data['X_test']
    Y_test = data['Y_test']

    model = NoteClassifier()
    model(X_train[:1])  # build model by calling it once
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(X_train, Y_train, batch_size=32, epochs=20, validation_data=(X_test, Y_test))
