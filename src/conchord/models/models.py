from pathlib import Path

import numpy as np
from keras import Model
from keras.callbacks import ReduceLROnPlateau
from keras.layers import (
    Add,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
    ReLU,
    TimeDistributed,
)
from keras.losses import CategoricalCrossentropy
from keras.metrics import Precision, Recall
from keras.optimizers.legacy import Adam
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow_addons.losses import SigmoidFocalCrossEntropy

from src.conchord.utils.parser import get_model_parser


def identity_block(X, filters):
    shortcut = X

    X = Conv2D(filters, (1, 1), padding='same')(X)
    X = ReLU()(X)

    X = Conv2D(filters, (1, 3), padding='same')(X)
    X = ReLU()(X)

    X = Conv2D(filters, (1, 1), padding='same')(X)

    X = Add()([X, shortcut])
    X = ReLU()(X)

    return X


def build_classifier(num_classes: int, calssifier_type: str, input_shape=(22, 12, 1), base_scale=2**4):
    inputs = Input(shape=input_shape)

    X = Conv2D(filters=base_scale, kernel_size=(4, 4), padding='same')(inputs)
    X = BatchNormalization()(X)
    X = ReLU()(X)
    X = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='same')(X)

    X = Conv2D(filters=base_scale * 2, kernel_size=(2, 2), padding='same')(X)
    X = BatchNormalization()(X)
    X = ReLU()(X)

    X = Conv2D(filters=base_scale * 2**2, kernel_size=(2, 2), padding='same')(X)
    X = BatchNormalization()(X)
    X = ReLU()(X)

    X = Conv2D(filters=base_scale * 2**3, kernel_size=(2, 2), padding='same')(X)
    X = BatchNormalization()(X)
    X = ReLU()(X)

    X = identity_block(X, filters=base_scale * 2**3)

    X = AveragePooling2D(pool_size=(1, 2), strides=(1, 2), padding='same')(X)

    X = TimeDistributed(Flatten())(X)
    X = TimeDistributed(Dense(base_scale * 2**3, activation='relu'))(X)
    X = Dropout(0.5)(X)

    activation = 'softmax' if calssifier_type == 'chords' else 'sigmoid'
    outputs = TimeDistributed(Dense(num_classes, activation=activation))(X)

    return Model(inputs=inputs, outputs=outputs)


def load_data(model_type: str, allowed_datasets: list[str] | None = None) -> dict[str, np.ndarray]:
    DATA_DIR = Path(__file__).resolve().parents[3] / 'data' / 'splitted'
    print(allowed_datasets)

    def load_split(split: str):
        data = np.load(str(DATA_DIR / f'{model_type}_{split}.npz'))
        X: np.ndarray = data['X']
        Y: np.ndarray = data['Y']
        sources: np.ndarray = data['sources']
        datasets: np.ndarray = data['datasets']
        if X.ndim == 3:
            X = np.expand_dims(X, axis=-1)
        if allowed_datasets is not None:
            mask = np.isin(datasets, allowed_datasets)
            X = X[mask]
            Y = Y[mask]
            sources = sources[mask]
            datasets = datasets[mask]

        return X, Y, sources, datasets

    X_train, Y_train, sources_train, datasets_train = load_split('train')
    X_val, Y_val, sources_val, datasets_val = load_split('val')
    X_test, Y_test, sources_test, datasets_test = load_split('test')
    num_classes = 128
    encoder = None

    if model_type == 'chords':
        Y_all = np.concatenate([Y_train.flatten(), Y_val.flatten(), Y_test.flatten()])
        num_classes = len(np.unique(Y_all))
        encoder = LabelEncoder()
        encoder.fit(Y_all)

        Y_train = encoder.transform(Y_train.flatten()).reshape(Y_train.shape)
        Y_val = encoder.transform(Y_val.flatten()).reshape(Y_val.shape)
        Y_test = encoder.transform(Y_test.flatten()).reshape(Y_test.shape)

        Y_train = to_categorical(Y_train, num_classes=num_classes)
        Y_val = to_categorical(Y_val, num_classes=num_classes)
        Y_test = to_categorical(Y_test, num_classes=num_classes)

        Y_train = Y_train.reshape((-1, 22, num_classes))
        Y_val = Y_val.reshape((-1, 22, num_classes))
        Y_test = Y_test.reshape((-1, 22, num_classes))

    return {
        'X_train': X_train,
        'Y_train': Y_train,
        'sources_train': sources_train,
        'datasets_train': datasets_train,
        'X_val': X_val,
        'Y_val': Y_val,
        'sources_val': sources_val,
        'datasets_val': datasets_val,
        'X_test': X_test,
        'Y_test': Y_test,
        'sources_test': sources_test,
        'datasets_test': datasets_test,
        'num_classes': num_classes,
        'encoder': encoder,
    }


if __name__ == '__main__':
    parser = get_model_parser()
    args = parser.parse_args()
    data = load_data(model_type=args.model_type, allowed_datasets=args.datasets)
    loss = SigmoidFocalCrossEntropy() if args.model_type == 'notes' else CategoricalCrossentropy()

    model: Model = build_classifier(num_classes=data['num_classes'], calssifier_type=args.model_type)
    lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss=loss,
        metrics=[
            'accuracy',
            #  BinaryAccuracy(name='bin_acc'),
            Precision(name='precision'),
            Recall(name='recall'),
        ],
    )
    model.summary()
    model.fit(
        data['X_train'],
        data['Y_train'],
        batch_size=128,
        epochs=args.epochs,
        validation_data=(data['X_val'], data['Y_val']),
        callbacks=[lr_schedule],
    )
    model.save(str(Path(__file__).resolve().parents[0] / f'{args.model_type}.keras'))
