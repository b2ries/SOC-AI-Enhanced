import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Model

class ThreatDetectionModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self.create_model()

    def create_model(self):
        input_layer = Input(shape=self.input_shape)
        x = LSTM(64, return_sequences=True)(input_layer)
        x = LSTM(32)(x)
        output_layer = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, epochs=10, validation_data=None):
        self.model.fit(X_train, y_train, epochs=epochs, validation_data=validation_data)

    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        return loss, accuracy

if __name__ == "__main__":
    # Exemple d'utilisation
    detection_model = ThreatDetectionModel((10, 100))  # 10 timesteps, 100 features
    detection_model.model.summary()
