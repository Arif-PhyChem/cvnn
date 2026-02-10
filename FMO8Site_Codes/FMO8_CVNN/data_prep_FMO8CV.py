import glob
import numpy as np
import torch

# Define data directory
data_dir = "D:/FMO_8site_data_init1/Training_data/*.npy"  # Adjust this to your actual directory

ostl_steps = 80
input_len = 81
n_states = 8
# Load all .npy files
file_list = glob.glob(data_dir)
# Function to process a single trajectory

labels = []
a = 1; b = n_states + 1
for i in range(0, n_states):
    for j in range(a, b):
        labels.append(j)
    a += n_states + 1 
    b += n_states 


print(labels)

def process_trajectory(file_path):
    traj = np.load(file_path)
    states = traj[0:401, labels]  #
    print(states.shape)
    for t in range(401):
        diag_idx = 0
        added_coeff = n_states
        for state in range(n_states):
            a = states[t, diag_idx].real
            states[t, diag_idx] = a + 1j * a  
            diag_idx += added_coeff
            added_coeff -= 1

    
    X_list = [states[k*ostl_steps:input_len + k*ostl_steps, :] for k in range(0, len(np.arange(0, 320, ostl_steps)))]
    Y_list = [states[input_len+k*ostl_steps:input_len+(k+1)*ostl_steps, :] for k in range(0, len(np.arange(0, 320, ostl_steps)))]
    #print(len(X_list))
    X = np.array(X_list)  # Shape: (4, 81, 36)
    Y = np.array(Y_list)  # Shape: (4, 80, 36)
    #print("X[-1] =", X[0][-1, 0])  # last ρ₁₁ in X
    #print("Y[0]  =", Y[0][0, 0])  # first ρ₁₁ in Y
    return X, Y

# Process all trajectories
all_X = []
all_Y = []
for file in file_list:
    X, Y = process_trajectory(file)
    all_X.append(X)
    all_Y.append(Y)

# Concatenate all data
all_X = np.concatenate(all_X, axis=0)  # Shape: (num_files * 4, 81, 36),
all_Y = np.concatenate(all_Y, axis=0)  # Shape:  (num_files * 4, 80, 36)
# Split into training and validation sets
num_samples = all_X.shape[0]
indices = np.random.permutation(num_samples)
train_size = int(0.8 * num_samples)
train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_X = all_X[train_indices]
train_Y = all_Y[train_indices]
val_X = all_X[val_indices]
val_Y = all_Y[val_indices]

# Save the processed NumPy arrays as .npy files
np.save('train_X.npy', train_X)
np.save('train_Y.npy', train_Y)
np.save('val_X.npy', val_X)
np.save('val_Y.npy', val_Y)

# Convert to PyTorch tensors
#train_X = torch.from_numpy(train_X).to(torch.complex64)
#train_Y = torch.from_numpy(train_Y).to(torch.complex64)
#val_X = torch.from_numpy(val_X).to(torch.complex64)
#val_Y = torch.from_numpy(val_Y).to(torch.complex64)

# The tensors train_X, train_Y, val_X, valY are now ready for SpinBosonCVPINN
