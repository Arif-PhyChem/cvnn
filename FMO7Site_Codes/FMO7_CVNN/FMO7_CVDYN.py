import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("E:/FMO_CVNN")
from FMO7CV_mlp import SpinBosonCVPINN, ComplexLinear, CRELU

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ostl_steps = 80
n_states = 7

# Load Model
model = torch.load(
    r"E:\FMO_CVNN\saved_models_FMOCVNew\fmo_model_epoch97687_vloss9.3229e-08.pth",
    map_location=device,
    weights_only=False
)
model = model.to(device)
model.eval()
print("Complex PINN full model loaded successfully.")

# Load ONE Test File
test_file = r"E:\7_sites_adolph_renger_H\test_data\init_1\7_initial-1_gamma-350.0_lambda-100.0_temp-270.0.npy"   # <--- change filename here
print("Running predictions on:", test_file)
test_data = np.load(test_file)

# Preprocess Data
labels = []
a, b = 1, n_states + 1
for i in range(n_states):
    for j in range(a, b):
        labels.append(j)
    a += n_states + 1
    b += n_states

test_states = test_data[0:401, labels]

for t in range(401):
    diag_idx = 0
    added_coeff = n_states
    for state in range(n_states):
        a = test_states[t, diag_idx].real
        test_states[t, diag_idx] = a + 1j * a
        diag_idx += added_coeff
        added_coeff -= 1

# Initial sequence (first 81 steps)
X_test = test_states[:81, :]
print("X TEST shape:", X_test.shape)

X_test_tensor = torch.tensor(
    X_test, dtype=torch.complex64
).unsqueeze(0).to(device)  # (1, 81, 28)

# Recursive Prediction
predicted_trajectory = []
X_curr = X_test_tensor

with torch.no_grad():
    for t in range(0, 320, ostl_steps):
        Y_pred = model(X_curr)
        predicted_trajectory.append(Y_pred.cpu().numpy().squeeze())
        X_curr = torch.cat([X_curr[:, ostl_steps:, :], Y_pred], dim=1)

predicted_trajectory = np.concatenate(predicted_trajectory, axis=0)
print("Predicted trajectory shape:", predicted_trajectory.shape)

# Save Results
save_path = f"initial_and_predicted_trajectory_{os.path.basename(test_file)}"
np.save(save_path, {"initial": X_test, "predicted": predicted_trajectory})
print(f"Saved results to {save_path}")

# Plotting
diag_indices = []
offset = 0
for i in range(n_states):
    diag_indices.append(offset)
    offset += (n_states - i)

colors = ['r', 'g', 'b', 'k', 'darkorange', 'brown', 'purple']
time_initial = np.linspace(0, 81 / 400, 81)
time_predicted = np.linspace(81 / 400, 1.0, 320)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

# Diagonal elements
for i, idx in enumerate(diag_indices):
    color = colors[i % len(colors)]
    ax1.plot(time_initial, X_test[:, idx].real, '--', linewidth=4, color=color, alpha=0.9)
    ax1.plot(time_predicted, predicted_trajectory[:, idx].real, '-', linewidth=4.5, color=color, alpha=1.0)
    ax1.plot(time_predicted, test_states[81:401, idx].real, linewidth=0.0, color=color,
             marker='o', markevery=10, ms=10, alpha=0.9)

ax1.set_title("FMO: CVNN", fontsize=45)
ax1.grid(True)

# Legend for diagonals
diag_handles = [
    plt.Line2D([0], [0], color='r', linewidth=4, linestyle='-', label=r"$\rho_{11}$"),
    plt.Line2D([0], [0], color='g', linewidth=4, linestyle='-', label=r"$\rho_{22}$"),
    plt.Line2D([0], [0], color='b', linewidth=4, linestyle='-', label=r"$\rho_{33}$"),
    plt.Line2D([0], [0], color='k', linewidth=4, linestyle='-', label=r"$\rho_{44}$"),
    plt.Line2D([0], [0], color='darkorange', linewidth=4, linestyle='-', label=r"$\rho_{55}$"),
    plt.Line2D([0], [0], color='brown', linewidth=4, linestyle='-', label=r"$\rho_{66}$"),
    plt.Line2D([0], [0], color='purple', linewidth=4, linestyle='-', label=r"$\rho_{77}$"),
]
ax1.legend(handles=diag_handles, fontsize=25, ncol=1, loc='upper right')
ax1.set_ylim(0.0, 1.05)

# --- Off-diagonal elements ---
off_diag_indices = {"ρ₁₂": 1, "ρ₂₃": 8, "ρ₃₄": 14}
re_colors = {"ρ₁₂": 'r', "ρ₂₃": 'g', "ρ₃₄": 'b'}
im_colors = {"ρ₁₂": 'darkorange', "ρ₂₃": 'purple', "ρ₃₄": 'brown'}

for label, idx in off_diag_indices.items():
    ax2.plot(time_initial, X_test[:, idx].real, '--', color=re_colors[label], linewidth=4, alpha=0.9)
    ax2.plot(time_predicted, predicted_trajectory[:, idx].real, '-', color=re_colors[label], linewidth=4.5, alpha=1.0)
    ax2.plot(time_predicted, test_states[81:401, idx].real, color=re_colors[label], linewidth=0.0,
             marker='o', markevery=10, ms=10, alpha=0.9)

    ax2.plot(time_initial, X_test[:, idx].imag, '--', color=im_colors[label], linewidth=4, alpha=0.9)
    ax2.plot(time_predicted, predicted_trajectory[:, idx].imag, '-', color=im_colors[label], linewidth=4.5, alpha=1.0)
    ax2.plot(time_predicted, test_states[81:401, idx].imag, color=im_colors[label], linewidth=0.0,
             marker='o', markevery=10, ms=10, alpha=0.9)

ax2.set_xlabel("Time (ps)", fontsize=40)
ax2.grid(True)

# Legend for off-diagonals (solid line only)
offdiag_handles = [
    plt.Line2D([0], [0], color='r', linewidth=4, linestyle='-', label=r"Re($\rho_{12}$)"),
    plt.Line2D([0], [0], color='darkorange', linewidth=4, linestyle='-', label=r"Im($\rho_{12}$)"),
    plt.Line2D([0], [0], color='g', linewidth=4, linestyle='-', label=r"Re($\rho_{23}$)"),
    plt.Line2D([0], [0], color='purple', linewidth=4, linestyle='-', label=r"Im($\rho_{23}$)"),
    plt.Line2D([0], [0], color='b', linewidth=4, linestyle='-', label=r"Re($\rho_{34}$)"),
    plt.Line2D([0], [0], color='brown', linewidth=4, linestyle='-', label=r"Im($\rho_{34}$)"),
]
ax2.legend(handles=offdiag_handles, fontsize=25, ncol=1, loc='upper right')

# Ticks & layout
ax1.tick_params(axis='both', which='major', labelsize=30)
ax2.tick_params(axis='both', which='major', labelsize=30)
plt.tight_layout(pad=1.5)

plt.savefig(f"combined_diagonal_offdiagonal_{os.path.basename(test_file)}.png",
            dpi=300, bbox_inches='tight')
plt.show()

