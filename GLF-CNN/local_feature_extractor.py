import os
import numpy as np
import scipy.io as sio
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, ELU, LayerNormalization

class TimeDomainFeatureExtractor(tf.keras.Model):
    def __init__(self):
        super(TimeDomainFeatureExtractor, self).__init__()
        self.conv1 = Conv1D(filters=128, kernel_size=10, strides=6, input_shape=(None, 10))
        self.elu = ELU()
        self.layer_norm = LayerNormalization(axis=[1, 2])

    def call(self, x):
        x = self.conv1(x)
        x = self.elu(x)
        x = self.layer_norm(x)
        return x

def extract_features(data):
    model = TimeDomainFeatureExtractor()
    return model(data)

def load_preprocessed_data(file_path):
    mat = sio.loadmat(file_path)
    return mat['segmented_signal']

def main():
    base_dir = r'C:\Users\Hp\Desktop\EMG models\ninapro db1'

    for i in range(1, 28):
        subject_dir = os.path.join(base_dir, f's{i}', f's{i}_filtered')
        new_directory = os.path.join(base_dir, f's{i}', f's{i}_local_features')
        if not os.path.exists(new_directory):
            os.makedirs(new_directory)

        for j in range(1, 4):
            file_path = os.path.join(subject_dir, f's{i}_A1_E{j}_filtered.mat')
            data = load_preprocessed_data(file_path)
            data_tensor = tf.convert_to_tensor(data, dtype=tf.float32)
            data_tensor = tf.transpose(data_tensor, perm=[0, 2, 1])
            features = extract_features(data_tensor)
            
            # Save the features
            output_file_path = os.path.join(new_directory, f's{i}_A1_E{j}_local_features.mat')
            sio.savemat(output_file_path, {'local_features': features.numpy()})
            print(f"Local features saved to {output_file_path}")

if __name__ == "__main__":
    main()
