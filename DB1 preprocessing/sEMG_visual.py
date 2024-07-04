import scipy.io
import matplotlib.pyplot as plt
import os

# Base directory containing the data of subjects from s1, s2, ... upto s27 of Ninapro DB1
base_dir = r'C:\Users\Hp\Desktop\EMG models\ninapro db1'

# Generate the list of file paths
file_paths = []
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.mat'):
            file_paths.append(os.path.join(root, file))

# Function to plot sEMG signals from a .mat file
def plot_emg_signals(file_path):
    mat = scipy.io.loadmat(file_path)
    
    data_key = 'emg'
    if data_key not in mat:
        raise KeyError(f"Key '{data_key}' not found in the .mat file")

    emg_data = mat[data_key]

    # Check the shape of the data to ensure it has 10 channels
    if emg_data.shape[1] != 10:
        raise ValueError("The EMG data does not have 10 channels")

    # Plot the EMG data
    fig, axs = plt.subplots(10, 1, figsize=(15, 20), sharex=True)
    fig.suptitle(f'EMG Signals from 10 Channels - {os.path.basename(file_path)}', fontsize=16)

    time = range(emg_data.shape[0])  # Each row is a time point

    for i in range(10):
        axs[i].plot(time, emg_data[:, i])
        axs[i].set_title(f'Channel {i + 1}')
        axs[i].set_ylabel('Amplitude')

    axs[-1].set_xlabel('Time (samples)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Plot EMG signals for each file
for file_path in file_paths:
    plot_emg_signals(file_path)
