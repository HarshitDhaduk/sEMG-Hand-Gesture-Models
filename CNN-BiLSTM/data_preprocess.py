import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
import pywt

# Define the data directory and parameters
data_dir = r'C:\Users\Hp\Desktop\EMG Models\ninapro db1'
NT = 500  # Target number of samples per trial
NC = 10  # Number of channels for NinaPro DB1
SL = 25  # Number of sub-sequences
TS = 20  # Time steps in each sub-sequence

# Step 1: Load Data
def load_data(data_dir):
    data = []
    for subj in range(1, 28):
        subj_folder = os.path.join(data_dir, f's{subj}')
        for trial_num in range(1, 4):
            file_name = f'S{subj}_A1_E{trial_num}.mat'
            file_path = os.path.join(subj_folder, file_name)
            mat_data = loadmat(file_path)
            # Assuming the EMG data is stored under the key 'emg'
            emg_data = mat_data['emg']
            # Segment the data into trials of 500 samples each
            num_trials = emg_data.shape[0] // NT
            trials = [emg_data[i*NT:(i+1)*NT] for i in range(num_trials)]
            data.extend(trials)
    return np.array(data)

data = load_data(data_dir)

# Step 2: Apply Wavelet Denoising
def wavelet_denoising(data, wavelet='sym8', level=8):
    denoised_data = []
    for trial in data:
        denoised_trial = []
        for channel in trial.T:
            coeffs = pywt.wavedec(channel, wavelet, level=level)
            threshold = np.sqrt(2 * np.log(len(channel))) * np.median(np.abs(coeffs[-level])) / 0.6745
            denoised_channel = pywt.waverec([pywt.threshold(c, value=threshold, mode='soft') for c in coeffs], wavelet)
            denoised_trial.append(denoised_channel[:len(channel)])  # Truncate to original length if necessary
        denoised_data.append(np.array(denoised_trial).T)
    return np.array(denoised_data)

data = wavelet_denoising(data)

# Step 3: Zero Padding / Truncating
def adjust_trial_length(data, NT):
    adjusted_data = []
    for trial in data:
        if trial.shape[0] > NT:
            adjusted_trial = trial[:NT]
        else:
            adjusted_trial = np.pad(trial, ((0, NT - trial.shape[0]), (0, 0)), 'constant')
        adjusted_data.append(adjusted_trial)
    return np.array(adjusted_data)

data = adjust_trial_length(data, NT)

# Step 4: Standardization
def standardize_data(data):
    mean = np.mean(data, axis=(0, 1))
    std = np.std(data, axis=(0, 1))
    standardized_data = (data - mean) / std
    return standardized_data

data = standardize_data(data)

# Step 5: Reshape Data
def reshape_data(data, SL, TS, NC):
    reshaped_data = []
    for trial in data:
        reshaped_trial = trial.reshape(SL, TS, NC)
        reshaped_data.append(reshaped_trial)
    return np.array(reshaped_data)

data = reshape_data(data, SL, TS, NC)

# Step 6: Split Data into Train and Test Sets
def split_data(data, train_ratio=0.7):
    N = data.shape[0]
    indices = np.arange(N)
    np.random.shuffle(indices)
    train_size = int(train_ratio * N)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    return data[train_indices], data[test_indices]

train_data, test_data = split_data(data)

# Convert to pandas DataFrames
train_df = pd.DataFrame([trial.flatten() for trial in train_data])
test_df = pd.DataFrame([trial.flatten() for trial in test_data])

# Save to files or proceed with TensorFlow training
train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)
