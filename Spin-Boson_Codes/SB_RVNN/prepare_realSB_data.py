import glob
import numpy as np
import torch
import os

# Training Data directory
data_dir = "E:/data/total_data/Training_data/combined/*.npy"

#Loading npy files

file_list = glob.glob(data_dir)
ostl_steps = 40
input_len = 81


# Process a single file, so defining function for it
def process_trajectory(file_path):
    traj = np.load(file_path)  # Shape: (401, 5), complex
    states = traj[:, [1, 2, 4]]  # [ρ11, ρ12, ρ22]
    # real-valued features
    real_states = np.zeros((401, 4), dtype=np.float32)
    real_states[:, 0] = states[:, 0].real  # Re(ρ11)
    real_states[:, 1] = states[:, 1].real  # Re(ρ12)
    real_states[:, 2] = states[:, 1].imag  # Im(ρ12)
    real_states[:, 3] = states[:, 2].real  # Re(ρ22)
    print(real_states)

    # Create training samples
    X_list = [real_states[k * ostl_steps:input_len + k * ostl_steps, :] for k in
              range(0, len(np.arange(0, 320, ostl_steps)))]
    Y_list = [real_states[input_len + k * ostl_steps:input_len + (k + 1) * ostl_steps, :] for k in
              range(0, len(np.arange(0, 320, ostl_steps)))]


    X = np.array(X_list)
    Y = np.array(Y_list)
    return X, Y

# Process all trajectories
all_X = []
all_Y = []
for file in file_list:
    X, Y = process_trajectory(file)
    all_X.append(X)
    all_Y.append(Y)

# Concatenate all data
all_X = np.concatenate(all_X, axis=0)  # Shape: (num_trajectories * 320, 81, 4)
all_Y = np.concatenate(all_Y, axis=0)  # Shape: (num_trajectories * 320, 4)

# === Optional Normalization (commented out) ===
# mean = np.mean(all_X, axis=(0, 1))
# std = np.std(all_X, axis=(0, 1))
# all_X = (all_X - mean) / std
# all_Y = (all_Y - mean) / std


# === Train/Validation Split ===
num_samples = all_X.shape[0]
indices = np.random.permutation(num_samples)
train_size = int(0.8 * num_samples)

train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_X = all_X[train_indices]
train_Y = all_Y[train_indices]
val_X = all_X[val_indices]
val_Y = all_Y[val_indices]

print(f" Training samples: {train_X.shape[0]}")
print(f" Validation samples: {val_X.shape[0]}")

# Save datasets
os.makedirs("data", exist_ok=True)
np.save('train_X.npy', train_X)
np.save('train_Y.npy', train_Y)
np.save('val_X.npy', val_X)
np.save('val_Y.npy', val_Y)
print(" All data saved as .npy files.")


# Convert to PyTorch tensors
#train_X_real = torch.tensor(train_X, dtype=torch.float32)
#train_Y_real = torch.tensor(train_Y, dtype=torch.float32)
#val_X_real = torch.tensor(val_X, dtype=torch.float32)
#val_Y_real = torch.tensor(val_Y, dtype=torch.float32)
#
#print(f"Train X shape: {train_X.shape}, Train Y shape: {train_Y.shape}")
#print(f"Val X shape: {val_X.shape}, Val Y shape: {val_Y.shape}")

