import os
import numpy as np
import scipy.io
import pywt
import matplotlib.pyplot as plt

# Base directory containing the folders s1, s2, ..., s27
base_dir = r'C:\Users\Hp\Desktop\EMG models\ninapro db1'

# Generate the list of file paths
file_paths = []
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.mat'):
            file_paths.append(os.path.join(root, file))

# Normalization function
def normalize(data):
    normalized_data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    return normalized_data

# Wavelet denoising function
def wavelet_denoising(data, wavelet='sym8', level=8):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-level])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    coeffs[1:] = [pywt.threshold(i, value=uthresh, mode='soft') for i in coeffs[1:]]
    denoised_data = pywt.waverec(coeffs, wavelet)
    return denoised_data

# Signal segmentation function
def segment_signal(data, window_size=200, step_size=20):
    num_samples, num_channels = data.shape
    segments = []
    for start in range(0, num_samples - window_size + 1, step_size):
        end = start + window_size
        segment = data[start:end, :]
        segments.append(segment)
    return np.array(segments)

# Process each file
for file_path in file_paths:
    # Load the .mat file
    mat = scipy.io.loadmat(file_path)
    
    # Using data variable emg for signal processing
    data_key = 'emg'  # replace with the actual key if different
    if data_key not in mat:
        print(f"Key '{data_key}' not found in {file_path}")
        continue
    signal = mat[data_key]

    # Apply normalization to the signal
    normalized_signal = normalize(signal)

    # Apply denoising to the signal
    denoised_signal = np.apply_along_axis(wavelet_denoising, 0, normalized_signal, wavelet='sym8', level=8)

    # Segment the signal
    window_size = 250  # 100 ms * 2L
    step_size = 25    # 10 ms * 2L
    segmented_signal = segment_signal(denoised_signal, window_size=window_size, step_size=step_size)

    # Determine the new directory path for saving the filtered data
    directory, filename = os.path.split(file_path)
    base_folder = os.path.basename(directory)
    new_directory = os.path.join(directory, f"{base_folder}_filtered")

    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    # Save the segmented signal to a new .mat file
    output_file_path = os.path.join(new_directory, f"{os.path.splitext(filename)[0]}_filtered.mat")
    scipy.io.savemat(output_file_path, {'segmented_signal': segmented_signal})
    print(f"Processed data saved to {output_file_path}")
