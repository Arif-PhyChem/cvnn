import glob
import numpy as np
import torch

# Define data directory
data_dir = "D:/4_sites_hypothetical/4_sites_hypothetical/training_data/*.npy"

# Load all .npy files
file_list = glob.glob(data_dir)

ostl_steps = 80
input_len = 81
n_states = 4

labels = []
a, b = 0, n_states
for i in range(0, n_states):
    for j in range(a, b):
        labels.append(j)
    a += n_states + 1
    b += n_states

divider = n_states + 1
print(labels)

# Function to process a single trajectory (convert complex to real-valued)
def process_trajectory(file_path):
    traj = np.load(file_path)  # Shape: (2001, 17), complex
    print(traj.shape)
    states = traj[:401, 1:]  # Extract rho elements
    # Prepare real-valued features
    real_states = np.zeros((401, n_states ** 2))  # Shape: (401, 16)
    q = 0
    for p in labels:
        if p % divider == 0:
            real_states[:, q] = states[:, p].real
            q += 1
        else:
            real_states[:, q] = states[:, p].real
            q += 1
            real_states[:, q] = states[:, p].imag
            q += 1

    # Create training samples
    X_list = [real_states[k * ostl_steps:input_len + k * ostl_steps, :] for k in
              range(0, len(np.arange(0, 320, ostl_steps)))]
    Y_list = [real_states[input_len + k * ostl_steps:input_len + (k + 1) * ostl_steps, :] for k in
              range(0, len(np.arange(0, 320, ostl_steps)))]

    X = np.array(X_list)  # Shape: (4, 81, 16)
    Y = np.array(Y_list)  # Shape: (4, 80, 16)
    print(X.shape, Y.shape)
    return X, Y


# Process all trajectories
all_X = []
all_Y = []
for file in file_list:
    X, Y = process_trajectory(file)
    all_X.append(X)
    all_Y.append(Y)

# Concatenate all data
all_X = np.concatenate(all_X, axis=0)  # Shape: (1600, 81, 16)
all_Y = np.concatenate(all_Y, axis=0)  # Shape: (1600, 80, 16)
print(all_X.shape, all_Y.shape)

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
# train_X_real = torch.tensor(train_X, dtype=torch.float32)
# train_Y_real = torch.tensor(train_Y, dtype=torch.float32)
# val_X_real = torch.tensor(val_X, dtype=torch.float32)
# val_Y_real = torch.tensor(val_Y, dtype=torch.float32)
#
# print(f"Train X shape: {train_X.shape}, Train Y shape: {train_Y.shape}")
# print(f"Val X shape: {val_X.shape}, Val Y shape: {val_Y.shape}")

