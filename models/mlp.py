import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder


class KerasMLP:
    def __init__(self, lr, l2_penalty, n_epochs=50, input_shape=(4096,), n_classes=10, verbose=0):
        self.lr = lr
        self.l2_penalty = l2_penalty
        self.n_epochs = n_epochs
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.trained = False
        self.verbose = verbose
        self.build_model()

    def reset_weights(self):
        session = tf.keras.backend.get_session()
        for layer in self.model.layers:
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.initializer.run(session=session)
            if hasattr(layer, 'bias_initializer'):
                layer.bias.initializer.run(session=session)

    def build_model(self):
        if self.l2_penalty > 0:
            regularizer = tf.keras.regularizers.l2(self.l2_penalty)
        else:
            regularizer = None

        x = tf.keras.layers.Input(shape=self.input_shape)
        y = tf.keras.layers.Dense(units=self.n_classes, kernel_regularizer=regularizer, activation='softmax')(x)

        self.model = tf.keras.models.Model(x, y)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, epsilon=1e-8)

        """
        self.early_stopping_monitor = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            min_delta=0,
            patience=0,
            verbose=0,
            mode='auto',
            baseline=None,
            restore_best_weights=True
        )
        """

        self.model.compile(optimizer=self.optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def fit(self, Xtr, ytr, sample_weight=None):
        if self.trained:
            print('Warning: retraining already trained model. Resetting weights')
            self.reset_weights()
        _ytr = ytr.copy()
        if ytr.ndim < 2 or ytr.shape[1] != self.n_classes:
            onehot_encoder = OneHotEncoder(sparse=False)
            _ytr = onehot_encoder.fit_transform(_ytr.reshape(-1, 1))
        if sample_weight is not None:
            self.model.fit(x=Xtr,
                           y=_ytr,
                           epochs=self.n_epochs,
                           verbose=self.verbose,
                           sample_weight=sample_weight)
        else:
            self.model.fit(x=Xtr,
                           y=_ytr,
                           epochs=self.n_epochs,
                           verbose=self.verbose)
        self.trained = True

    def predict(self, X):
        probs = self.model(X)

        return tf.keras.backend.get_value(probs).argmax(axis=1) + 1

    def predict_proba(self, X):
        probs = self.model(X)

        return tf.keras.backend.get_value(probs)
