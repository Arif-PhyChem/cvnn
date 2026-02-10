import glob
import numpy as np
import torch

# Define data directory
data_dir = "E:/data/total_data/Training_data/combined/*.npy"  # Adjust this to your actual directory

ostl_steps = 40 
input_len = 81

# Load all .npy files
file_list = glob.glob(data_dir)
# Function to process a single trajectory
def process_trajectory(file_path):
    traj = np.load(file_path)  # Shape: (401, 5), complex
    states = traj[:, [1, 2, 4]]  # [rho_11, rho_12, rho_22]
    for t in range(401):
        a = states[t, 0].real
        states[t, 0] = a + 1j * a
        b = states[t, 2].real
        states[t, 2] = b + 1j * b

    X_list = [states[k*ostl_steps:input_len + k*ostl_steps, :] for k in range(0, len(np.arange(0, 320, ostl_steps)))]
    Y_list = [states[input_len+k*ostl_steps:input_len+(k+1)*ostl_steps, :] for k in range(0, len(np.arange(0, 320, ostl_steps)))]
    #print(len(X_list))
    X = np.array(X_list)  # Shape: (8, 81, 3)
    Y = np.array(Y_list)  # Shape: (8, 3)
    return X, Y

# Process all trajectories
all_X = []
all_Y = []
for file in file_list:
    X, Y = process_trajectory(file)
    all_X.append(X)
    all_Y.append(Y)

# Concatenate all data
all_X = np.concatenate(all_X, axis=0)  # Shape: (num_trajectories * 320, 81, 3)
all_Y = np.concatenate(all_Y, axis=0)  # Shape: (num_trajectories * 320, 3)
print("Shap of All X and Y:", all_X.shape, all_Y.shape)
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