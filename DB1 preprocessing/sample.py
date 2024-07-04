import scipy.io
import numpy as np
from scipy.signal import butter, filtfilt

# Load the .mat file
file_path = "C:/Users/Hp/Desktop/EMG models/ninapro db1/s1/S1_A1_E1.mat"
mat_data = scipy.io.loadmat(file_path)

# Display the variable names in the .mat file
print(mat_data.keys())

# Get the variables
emg = mat_data['emg']
glove = mat_data['glove']
stimulus = mat_data['stimulus']

# Interpolation function
def interpolate_to_match(emg, glove, glove_sampling_rate, emg_sampling_rate):
    n_samples_emg = emg.shape[0]
    n_samples_glove = glove.shape[0]
    x_old = np.linspace(0, n_samples_emg / emg_sampling_rate, n_samples_emg)
    x_new = np.linspace(0, n_samples_emg / emg_sampling_rate, n_samples_glove)
    
    glove_interpolated = np.zeros((n_samples_emg, glove.shape[1]))
    
    for i in range(glove.shape[1]):
        glove_interpolated[:, i] = np.interp(x_old, x_new, glove[:, i])
        
    return glove_interpolated

# Interpolating glove data to match the length of EMG data
glove_interpolated = interpolate_to_match(emg, glove, glove_sampling_rate=25, emg_sampling_rate=100)

# Low-pass filtering EMG signals
def butter_lowpass_filter(data, cutoff, fs, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data, axis=0)
    return y

# Applying the filter
cutoff_frequency = 5  # Hz
emg_filtered = butter_lowpass_filter(emg, cutoff=cutoff_frequency, fs=100)

# Relabeling strategy using GLR
def relabel_emg_glr(emg, stimulus, window_size=100):
    labels = np.zeros_like(stimulus)
    
    for i in range(0, len(emg) - window_size, window_size):
        window = emg[i:i + window_size]
        # Whitening the signal
        window_mean = np.mean(window, axis=0)
        window_whitened = window - window_mean

        # GLR: Optimal change points detection
        glr = np.zeros(window_size)
        for t in range(1, window_size):
            mu1 = np.mean(window_whitened[:t], axis=0)
            mu2 = np.mean(window_whitened[t:], axis=0)
            sigma1 = np.var(window_whitened[:t], axis=0)
            sigma2 = np.var(window_whitened[t:], axis=0)
            glr[t] = np.sum((window_whitened[:t] - mu1)**2 / sigma1) + np.sum((window_whitened[t:] - mu2)**2 / sigma2)
        
        # Find change points with significant GLR
        change_points = np.where(glr > np.mean(glr) + 2 * np.std(glr))[0]
        
        if len(change_points) > 0:
            labels[i + change_points[0]:i + change_points[-1]] = 1
    
    return labels

# Applying relabeling strategy
relabels_glr = relabel_emg_glr(emg_filtered, stimulus)
mat_data['relabels_glr'] = relabels_glr

# Storing the filtered EMG signals back
mat_data['emg_filtered'] = emg_filtered

# Optionally, save the modified data to a new .mat file
scipy.io.savemat("C:/Users/Hp/Desktop/EMG models/ninapro db1/s1/S1_A1_E1_filtered.mat", mat_data)
