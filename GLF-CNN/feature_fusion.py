import os
import numpy as np
import scipy.io as sio
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Flatten, Dense, ELU, LayerNormalization, Dropout, DepthwiseConv1D, Concatenate

class DepthWidthSeparableConvLayer(tf.keras.layers.Layer):
    def __init__(self, depth_multiplier=1):
        super(DepthWidthSeparableConvLayer, self).__init__()
        self.depthwise_conv = DepthwiseConv1D(kernel_size=3, depth_multiplier=depth_multiplier, padding='same')
        self.pointwise_conv = Conv1D(filters=128, kernel_size=1, padding='same')

    def call(self, inputs):
        x = self.depthwise_conv(inputs)
        x = self.pointwise_conv(x)
        return x

class FeatureFusionModel(tf.keras.Model):
    def __init__(self):
        super(FeatureFusionModel, self).__init__()
        self.depthwidth_separable_conv = DepthWidthSeparableConvLayer()
        self.global_avg_pool = GlobalAveragePooling1D()
        self.flatten = Flatten()
        self.fc1 = Dense(128, activation=None)
        self.elu1 = ELU()
        self.layer_norm1 = LayerNormalization()
        self.dropout1 = Dropout(0.2)
        self.fc2 = Dense(512, activation=None)
        self.elu2 = ELU()
        self.layer_norm2 = LayerNormalization()
        self.dropout2 = Dropout(0.2)
        self.fc3 = Dense(512, activation=None)
        self.elu3 = ELU()
        self.layer_norm3 = LayerNormalization()
        self.fc_output = Dense(10, activation='softmax')  # Adjust the number of neurons based on the number of classes

    def call(self, local_features, global_features):
        combined = local_features + global_features
        combined_conv = self.depthwidth_separable_conv(combined)
        combined += combined_conv
        x = self.global_avg_pool(combined)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.elu1(x)
        x = self.layer_norm1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.elu2(x)
        x = self.layer_norm2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.elu3(x)
        x = self.layer_norm3(x)
        output = self.fc_output(x)
        return output

def load_features(file_path, feature_key):
    mat = sio.loadmat(file_path)
    return mat[feature_key]

def main():
    base_dir = r'C:\Users\Hp\Desktop\EMG models\ninapro db1'

    all_local_features = []
    all_global_features = []

    for i in range(1, 28):
        local_dir = os.path.join(base_dir, f's{i}', f's{i}_local_features')
        global_dir = os.path.join(base_dir, f's{i}', f's{i}_global_features')

        for j in range(1, 4):
            local_file_path = os.path.join(local_dir, f's{i}_A1_E{j}_local_features.mat')
            global_file_path = os.path.join(global_dir, f's{i}_A1_E{j}_global_features.mat')

            local_features = load_features(local_file_path, 'local_features')
            global_features = load_features(global_file_path, 'global_features')

            all_local_features.append(local_features)
            all_global_features.append(global_features)

    all_local_features = np.concatenate(all_local_features, axis=0)
    all_global_features = np.concatenate(all_global_features, axis=0)

    local_features_tensor = tf.convert_to_tensor(all_local_features, dtype=tf.float32)
    global_features_tensor = tf.convert_to_tensor(all_global_features, dtype=tf.float32)

    fusion_model = FeatureFusionModel()
    output = fusion_model(local_features_tensor, global_features_tensor)

    # Save the final output or predictions as needed
    # Example:
    output_file_path = os.path.join(base_dir, 'final_output.mat')
    sio.savemat(output_file_path, {'output': output.numpy()})
    print(f"Final output saved to {output_file_path}")

if __name__ == "__main__":
    main()
