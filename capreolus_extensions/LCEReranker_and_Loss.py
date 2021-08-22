import tensorflow as tf
from tensorflow.keras.layers import Layer


class KerasLCEModel(tf.keras.Model):
    def __init__(self, model, *args, **kwargs):
        super(KerasPairModel, self).__init__(*args, **kwargs)
        self.model = model

    def call(self, x, **kwargs):
        # TODO
        pass

    def predict_step(self, data):
        # TODO
        pass


class TFLCELoss:
    def call(self):
        # TODO:
        pass
