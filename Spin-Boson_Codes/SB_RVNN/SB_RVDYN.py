import os
# --- Avoid OpenMP duplicate runtime crash
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch
from my_real_mlp import SpinBosonRealPINN, RealLinear
import matplotlib.pyplot as plt

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config
ostl_steps = 40
T_SEED = 81
HORIZON = 320
DT = 0.05  # time step

# Load the trained model
MODEL_PATH = r"E:\sb\real_mlp\saved_models\real_mlp_model-261955-tloss-3.2351e-07-vloss-6.1471e-07.pth"
model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.eval()
print("Full model loaded successfully.")

# Test trajectory file
TEST_FILE = r"E:\data\total_data\test_set\2_epsilon-0.0_Delta-1.0_lambda-0.6_gamma-8.0_beta-1.0.npy"

# Process a single test trajectory
def process_test_trajectory(file_path):
    traj = np.load(file_path)  # Shape: (401, 5), complex
    states = traj[:, [1, 2, 4]]  # Extract [ρ11, ρ12, ρ22]

    # Keep SAME order as training: [ρ11, Re(ρ12), Im(ρ12), ρ22]
    real_states = np.zeros((401, 4))
    real_states[:, 0] = states[:, 0].real  # ρ11
    real_states[:, 1] = states[:, 1].real  # Re(ρ12)
    real_states[:, 2] = states[:, 1].imag  # Im(ρ12)
    real_states[:, 3] = states[:, 2].real  # ρ22
    return real_states

# Load and preprocess test data
test_data = process_test_trajectory(TEST_FILE)

# Initial seed (first 81 steps)
X_seed = test_data[:T_SEED, :]
X_seed_tensor = torch.tensor(X_seed, dtype=torch.float32).unsqueeze(0).to(device)  # (1,81,4)
print(f"Test seed shape: {X_seed_tensor.shape}")

# Ground truth for comparison
GT_future = test_data[T_SEED:T_SEED + HORIZON, :]

# Recursive prediction
preds = []
X_curr = X_seed_tensor.clone()

with torch.no_grad():
    for t in range(0, HORIZON, ostl_steps):  # predict horizon = 320
        Y_pred = model(X_curr)  # (1, 40, 4)
        preds.append(Y_pred.cpu().numpy().squeeze())  # (40,4)
        # Slide the window: drop 40 steps, append prediction
        X_curr = torch.cat([X_curr[:, ostl_steps:, :], Y_pred], dim=1)

# Concatenate predictions
pred = np.concatenate(preds, axis=0)  # (320, 4)

# Add back the seed at the front
predicted_full = np.concatenate((X_seed, pred), axis=0)  # (401, 4)

# Save results
np.save("initial_and_predicted_trajectory_real.npy",
        {"initial": X_seed, "predicted": pred})
print("Saved predicted trajectory to 'initial_and_predicted_trajectory_real.npy'")

# --- Plot ---
legend_labels = [r'$\rho_{11}$', r'Re($\rho_{12}$)', r'Im($\rho_{12}$)', r'$\rho_{22}$']
colors = ["red", "green", "orange", "blue"]

time_ps = np.arange(T_SEED + HORIZON) * DT

plt.figure(figsize=(12, 12))
for i in range(4):
    c = colors[i]
    # initial seed
    plt.plot(time_ps[:T_SEED], X_seed[:, i],
             linestyle='--', linewidth=4, color=c, label='_nolegend_')
    # predictions (solid line)
    plt.plot(time_ps[T_SEED:T_SEED+HORIZON], pred[:, i],
             linestyle='-', linewidth=4.5, color=c, label=legend_labels[i])
    # ground truth (dashed line + markers)
    plt.plot(time_ps[T_SEED:T_SEED+HORIZON], GT_future[:, i],
              linewidth=0.0, color=c,
             marker='o', markersize=10, markevery=10, alpha=0.9, label='_nolegend_')

plt.title("SB: RVNN", fontsize=45)
plt.xlabel(r"Time ($1/\Delta$)", fontsize=35)
plt.grid(True)
plt.legend(fontsize=25, loc='upper right', ncol=1)
plt.tick_params(axis='both', which='major', labelsize=30)
#plt.margins(x=0.02, y=0.0)
#plt.tight_layout()

out_png_path = "prediction_vs_actual_rvnn.png"
plt.savefig(out_png_path, dpi=300)
print(f" Saved plot: {out_png_path}")

plt.show()
plt.close()
