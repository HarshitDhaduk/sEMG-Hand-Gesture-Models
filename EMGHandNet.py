import os
import numpy as np
import scipy.io as sio
from scipy.io import savemat
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Bidirectional, LSTM, Dense, TimeDistributed, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.utils import to_categorical

# Preprocess .mat file
def preprocess_mat_file(filepath):
    data = sio.loadmat(filepath)
    
    subject = data['subject']
    exercise = data['exercise']
    emg = data['emg']
    glove = data['glove']
    stimulus = data['stimulus']
    restimulus = data['restimulus']
    repetition = data['repetition']
    rerepetition = data['rerepetition']
    
    processed_data = {
        'subject': subject,
        'exercise': exercise,
        'emg': emg,
        'glove': glove,
        'stimulus': stimulus,
        'restimulus': restimulus,
        'repetition': repetition,
        'rerepetition': rerepetition
    }
    
    return processed_data

# Traverse directories and preprocess files
def traverse_and_process(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.mat'):
                file_path = os.path.join(root, file)
                processed_data = preprocess_mat_file(file_path)
                
                new_dir = os.path.join(root, 'EMGHandNet_filtered')
                if not os.path.exists(new_dir):
                    os.makedirs(new_dir)
                
                new_file_path = os.path.join(new_dir, file.replace('.mat', '_filtered.mat'))
                savemat(new_file_path, processed_data)

                print(f"Processed and saved: {new_file_path}")

# Data Preprocessing for EMGHandNet
def preprocess_data(sEMG_signals, labels, sampling_rate, time_duration, num_subjects, num_activities, num_repetitions):
    num_samples = num_subjects * num_activities * num_repetitions
    num_channels = sEMG_signals.shape[1]
    num_values = sampling_rate * time_duration

    data = np.zeros((num_samples, num_channels, num_values))
    labels = np.array(labels)
    
    for i in range(num_samples):
        for j in range(num_channels):
            data[i, j, :] = sEMG_signals[i, j, :num_values]
    
    def factorize_nt(nt):
        factors = [(i, nt // i) for i in range(1, int(nt**0.5)+1) if nt % i == 0]
        return factors[-1]
    
    num_subsequences, num_timesteps = factorize_nt(num_values)
    data_reshaped = np.reshape(data, (num_samples, num_subsequences, num_timesteps, num_channels))
    
    return data_reshaped, labels

# Build EMGHandNet Model
def emgHandNet_model(input_shape, num_activities):
    input_layer = Input(shape=input_shape)
    x = TimeDistributed(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))(input_layer)
    x = TimeDistributed(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))(x)
    x = TimeDistributed(Conv1D(filters=256, kernel_size=3, padding='same', activation='relu'))(x)
    x = TimeDistributed(Conv1D(filters=512, kernel_size=3, padding='same', activation='relu'))(x)

    x = TimeDistributed(Flatten())(x)
    x = Bidirectional(LSTM(units=128, return_sequences=True))(x)
    x = Bidirectional(LSTM(units=64, return_sequences=True))(x)
    x = Flatten()(x)

    x = Dense(units=512, activation='relu')(x)
    output_layer = Dense(units=num_activities, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
    
    return model

# Main function to run everything
if __name__ == "__main__":
    directory = r'C:\Users\Hp\Desktop\EMG models\ninapro db1'
    traverse_and_process(directory)
    
    # Actual sEMG signal and labels data of ninaPro DB1
    sEMG_signals = np.random.rand(100, 8, 4000)
    labels = np.random.randint(0, 5, 100)
    sampling_rate = 100
    time_duration = 5
    num_subjects = 27
    num_activities = 52
    num_repetitions = 2

    data_reshaped, labels = preprocess_data(sEMG_signals, labels, sampling_rate, time_duration, num_subjects, num_activities, num_repetitions)
    input_shape = data_reshaped.shape[1:]
    
    model = build_model(input_shape, num_activities)
    labels_categorical = to_categorical(labels, num_classes=num_activities)
    model.fit(data_reshaped, labels_categorical, epochs=10, batch_size=32, validation_split=0.2)

    loss, accuracy = model.evaluate(data_reshaped, labels_categorical)
    print(f"Model Loss: {loss}, Model Accuracy: {accuracy}")
