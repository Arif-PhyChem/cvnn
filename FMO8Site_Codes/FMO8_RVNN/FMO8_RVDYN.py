import numpy as np
import torch
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append("D:/FMO_8site_jia.etal/RVNN_8site")
from FMO8RV_mlp import SpinBosonRealPINN, RealLinear
import matplotlib.pyplot as plt

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ostl_steps = 80
n_states = 8

# Load the entire model
model = torch.load(
    r"D:\FMO_8site_jia.etal\RVNN_8site\saved_models_8siteRVNew\real_mlp_model-95823-tloss-1.0200e-07-vloss-6.6142e-07.pth",
    map_location=device,
    weights_only=False
)
model.eval()
print("Full model loaded successfully.")

# Pick ONE test file
test_file = r"D:\FMO_8site_data_init1\Test_data\8_initial-1_gamma-350.0_lambda-280.0_temp-90.0.npy"
print("Running predictions on:", test_file)

# Directory to save results
save_dir = "D:/FMO_8site_jia.etal/RVNN_8site/results"
os.makedirs(save_dir, exist_ok=True)


# Preprocessing
test_data = np.load(test_file)
print("test_data shape:", test_data.shape)

labels = []
a = 0
b = n_states
for i in range(n_states):
    for j in range(a, b):
        labels.append(j)
    a += n_states + 1
    b += n_states
divider = n_states + 1
print("Full Selected Labels:", labels)

diag_idx = []
idx = 0
for i in range(n_states):
    diag_idx.append(idx)
    adder = (n_states - (i + 1)) * 2
    idx += adder + 1
print("Diagonal indices:", diag_idx)

# Function to process a single trajectory
def process_test_trajectory(test_file):
    traj = np.load(test_file)
    states = traj[0:401, 1:]
    print("states shape:", states.shape)
    real_states = np.zeros((401, n_states**2))
    print("real_states shape:", real_states.shape)
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
    return states, real_states

complex_data, test_data = process_test_trajectory(test_file)


# Prepare input
X_test = test_data[:81, :]
print("X_test shape:", X_test.shape)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(0)
X_test_tensor = X_test_tensor.to(device)
print("X Tensor's Shape is:", X_test_tensor.shape)


initial_trajectory = X_test


# Recursive inference
predicted_trajectory = []
X_curr = X_test_tensor
with torch.no_grad():
    for t in range(0, 320, ostl_steps):
        Y_pred = model(X_curr)
        predicted_trajectory.append(Y_pred.cpu().numpy().squeeze())
        X_curr = torch.cat([X_curr[:, ostl_steps:, :], Y_pred], dim=1)

predicted_trajectory = np.concatenate(predicted_trajectory, axis=0)
time_step_ps = 0.0025

combined_trajectory = {
    "initial": initial_trajectory,
    "predicted": predicted_trajectory
}
combined_filename = os.path.join(
    save_dir,
    f"{os.path.basename(test_file).split('.')[0]}_initial_and_predicted_trajectory.npy"
)
np.save(combined_filename, combined_trajectory)
print(f"Saved combined trajectory as {combined_filename}")


# Plotting
colors = ['r', 'g', 'b', 'k', 'darkorange', 'brown', 'purple', 'lightseagreen']

fig, axs = plt.subplots(2, 1, figsize=(12, 12))

# Diagonals
ax1 = axs[0]
for i, idx in enumerate(diag_idx[:]):
    color = colors[i % len(colors)]
    ax1.plot(np.arange(81) * time_step_ps, X_test[:, idx].real, '--', linewidth=4, color=color, alpha=0.9)
    ax1.plot(np.arange(81, 401) * time_step_ps, predicted_trajectory[:, idx].real, '-', linewidth=4.5, color=color, alpha=1.0)
    ax1.plot(np.arange(81, 401) * time_step_ps, test_data[81:401, idx].real, linewidth=0.0, color=color,
             marker='o', markevery=10, ms=10, alpha=0.9)
    ax1.plot([], [], color=color, linestyle='-', linewidth=4.5, label=f"$\\rho_{{{i+1}{i+1}}}$")

ax1.set_title("FMO 8-Site: RVNN", fontsize=45)
ax1.tick_params(labelbottom=False)
ax1.grid(True)
#ax1.legend(fontsize=25, ncol=2, loc='upper right')

# Off-diagonals
ax2 = axs[1]
off_diag_indices = {"ρ₁₂": (1, 2), "ρ₂₃": (16, 17), "ρ₃₄": (29, 30)}
re_colors = {"ρ₁₂": 'r', "ρ₂₃": 'g', "ρ₃₄": 'b'}
im_colors = {"ρ₁₂": 'darkorange', "ρ₂₃": 'purple', "ρ₃₄": 'brown'}

for label, (re_idx, im_idx) in off_diag_indices.items():
    ax2.plot(np.arange(81) * time_step_ps, X_test[:, re_idx], '--', linewidth=4, color=re_colors[label], alpha=0.9)
    ax2.plot(np.arange(81, 401) * time_step_ps, predicted_trajectory[:, re_idx], '-', linewidth=4.5, color=re_colors[label], alpha=1.0)
    ax2.plot(np.arange(81, 401) * time_step_ps, test_data[81:401, re_idx], linewidth=0.0, color=re_colors[label],
             marker='o', markevery=10, ms=10, alpha=0.9)
    ax2.plot([], [], color=re_colors[label], linestyle='-', linewidth=4.5, label=f"Re({label})")

    ax2.plot(np.arange(81) * time_step_ps, X_test[:, im_idx], '--', linewidth=4, color=im_colors[label], alpha=0.9)
    ax2.plot(np.arange(81, 401) * time_step_ps, predicted_trajectory[:, im_idx], '-', linewidth=4.5, color=im_colors[label], alpha=1.0)
    ax2.plot(np.arange(81, 401) * time_step_ps, test_data[81:401, im_idx], linewidth=0.0, color=im_colors[label],
             marker='o', markevery=10, ms=10, alpha=0.9)
    ax2.plot([], [], color=im_colors[label], linestyle='-', linewidth=4.5, label=f"Im({label})")

ax2.set_xlabel("Time (ps)", fontsize=40)
ax2.grid(True)
#ax2.legend(fontsize=25, ncol=2, loc='lower right')

ax1.tick_params(axis='both', which='major', labelsize=30)
ax2.tick_params(axis='both', which='major', labelsize=30)
plt.tight_layout(pad=3)

plt.savefig(os.path.join(save_dir, f"rho_diag_and_offdiag_combined_{os.path.basename(test_file)}.png"))
plt.show()
