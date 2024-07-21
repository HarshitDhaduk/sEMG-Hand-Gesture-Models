import scipy.io

mat_path = 'ninapro db3\s1_0\DB3_s1\S1_E1_A1.mat'
mat = scipy.io.loadmat(mat_path)


for variable in mat:
    if variable.startswith('__'):
        continue
    data = mat[variable]
    print(f"{variable}")
    print(f"Type: {type(data)}")
    if hasattr(data, 'shape'):
        print(f"Shape: {data.shape}")
        if data.ndim > 1 and data.shape[0] > 5:
                print("First 5 rows:")
                print(data[:5])
        elif data.ndim == 1 and data.size > 5:
                print("First 5 elements:")
                print(data[:5])
        else:
                print("Data:")
                print(data)
    else:
        print("Data:")
        print(data)
    print()