import tensorflow as tf
from tensorflow.keras.layers import Conv1D, ELU, LayerNormalization

class TimeDomainFeatureExtractor(tf.keras.Model):
    def __init__(self):
        super(TimeDomainFeatureExtractor, self).__init__()
        self.conv1 = Conv1D(filters=128, kernel_size=10, strides=6, input_shape=(None, 10))
        self.elu = ELU()
        self.layer_norm = LayerNormalization(axis=[1, 2])  # Adjust axis based on your input length

    def call(self, x):
        x = self.conv1(x)
        x = self.elu(x)
        x = self.layer_norm(x)
        return x

# Function to extract features
def extract_features(data):
    model = TimeDomainFeatureExtractor()
    return model(data)
