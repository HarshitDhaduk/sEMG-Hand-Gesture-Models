import scipy.io

# Load the .mat file
file_path = "C:/Users/Hp/Desktop/EMG models/ninapro db1/s1/S1_A1_E1.mat"
mat_data = scipy.io.loadmat(file_path)

# Display the variable names in the .mat file
mat_variables = mat_data.keys()

print(mat_variables)