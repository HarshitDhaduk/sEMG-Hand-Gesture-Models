import numpy as np
from scipy.signal import butter, filtfilt

# Synchronization 
# Interpolate glove data to match EMG sampling rate (100Hz)

# Get the variables
emg = mat_data['emg']
glove = mat_data['glove']
stimulus = mat_data['stimulus']

# Interpolation function
def interpolate_to_match(emg, glove, glove_sampling_rate, emg_sampling_rate):
    n_samples_emg = len(emg)
    n_samples_glove = len(glove)
    x_old = np.linspace(0, n_samples_emg / emg_sampling_rate, n_samples_emg)
    x_new = np.linspace(0, n_samples_emg / emg_sampling_rate, n_samples_glove)
    glove_interpolated = np.interp(x_old, x_new, glove)
    return glove_interpolated

# Interpolating glove data to match the length of EMG data
glove_interpolated = interpolate_to_match(emg, glove, glove_sampling_rate=25, emg_sampling_rate=100)

# Low-pass filtering EMG signals using butterworth filter
def butter_lowpass_filter(data, cutoff, fs, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data, axis=0)
    return y

# Applying the filter
cutoff_frequency = 5  # Hz
emg_filtered = butter_lowpass_filter(emg, cutoff=cutoff_frequency, fs=100)

# Storing the filtered EMG signals back
mat_data['emg_filtered'] = emg_filtered
