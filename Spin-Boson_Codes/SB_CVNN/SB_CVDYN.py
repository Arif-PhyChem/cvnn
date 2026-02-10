import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from my_complex_mlp import SpinBosonCVPINN, ComplexLinear, CRELU

# --- Avoid OpenMP duplicate runtime crash ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# CONFIG
ostl_steps = 40
n_states = 2
dt = 0.05  # ps per step (1/Δ units)
MODEL_PATH = r"E:/sb/cv_nn/saved_models/cv_mlp_model-267803-tloss-2.7988e-07-vloss-6.2897e-07.pth"
TEST_FILE = r"E:\data\total_data\test_set\2_epsilon-0.0_Delta-1.0_lambda-0.6_gamma-9.0_beta-1.0.npy"

# Output directory
OUT_DIR = "E:/SB_CVNN/rollouts"
os.makedirs(OUT_DIR, exist_ok=True)

# Load Model
model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.eval()
print("Complex PINN model loaded successfully from:", MODEL_PATH)

# LOAD & PROCESS TEST DATA
test_data = np.load(TEST_FILE)  # (401, 5) complex
print("Shape of Test Data:", test_data.shape)

# Extract [ρ11, ρ12, ρ22]
test_states = test_data[:, [1, 2, 4]]  # (401, 3) complex

# Hermitian adjustment (force diagonals real)
for t in range(test_states.shape[0]):
    test_states[t, 0] = test_states[t, 0].real + 1j * test_states[t, 0].real  # ρ11
    test_states[t, 2] = test_states[t, 2].real + 1j * test_states[t, 2].real  # ρ22

# Initial seed (first 81 steps)
X_init = test_states[:81, :]  # (81, 3)
X_init_tensor = torch.tensor(X_init, dtype=torch.complex64).unsqueeze(0).to(device)
print("Initial seed shape:", X_init_tensor.shape)

# ROLL OUT PREDICTIONS
predicted_blocks = []
X_curr = X_init_tensor
with torch.no_grad():
    for _ in range(0, 320, ostl_steps):
        Y_pred = model(X_curr)  # (1, ostl_steps, 3)
        predicted_blocks.append(Y_pred.cpu().numpy().squeeze())
        X_curr = torch.cat([X_curr[:, ostl_steps:, :], Y_pred], dim=1)

# Concatenate
Y_pred_full = np.concatenate(predicted_blocks, axis=0).astype(np.complex128)  # (320, 3)
Y_true_seg = test_states[81:401, :]  # (320, 3)
print("Prediction shape:", Y_pred_full.shape)

# SAVE RESULTS
base = os.path.splitext(os.path.basename(TEST_FILE))[0]
sprefix = os.path.join(OUT_DIR, base)

out_npy = f"{sprefix}_initial_and_predicted_trajectory_complex.npy"
np.save(out_npy, {
    "initial": X_init,
    "predicted": Y_pred_full,
    "true_seg": Y_true_seg,
    "meta": {
        "dt": dt,
        "rho_columns": ["rho11", "rho12", "rho22"],
        "source_file": TEST_FILE,
        "model_path": MODEL_PATH,
        "ostl_steps": ostl_steps,
        "n_states": n_states,
        "initial_len": 81,
        "pred_len": 320
    }
})
print(f"Saved results: {out_npy}")

# Plotting my results

time_ps = np.arange(401) * dt
legend_labels = [r'$\rho_{11}$', r'Re($\rho_{12}$)', r'Im($\rho_{12}$)', r'$\rho_{22}$']
colors = ["red", "green", "orange", "blue"]

plt.figure(figsize=(12, 12))

for i in range(4):
    color = colors[i]

    if i < 3:
        # For rho_11, rho_22, Re(rho_12)
        if i == 0:  # rho_11
            seed_vals = X_init[:, 0].real
            pred_vals = Y_pred_full[:, 0].real
            true_vals = Y_true_seg[:, 0].real
        elif i == 1:  # rho_12
            seed_vals = X_init[:, 1].real
            pred_vals = Y_pred_full[:, 1].real
            true_vals = Y_true_seg[:, 1].real
        else:  # Re(rho_12)
            seed_vals = X_init[:, 1].imag
            pred_vals = Y_pred_full[:, 1].imag
            true_vals = Y_true_seg[:, 1].imag
    else:
        # Im(rho_22)
        seed_vals = X_init[:, 2].real
        pred_vals = Y_pred_full[:, 2].real
        true_vals = Y_true_seg[:, 2].real

    # Initial seed (dashed)
    plt.plot(time_ps[:81], seed_vals, linestyle='--', linewidth=4, color=color, label='_nolegend_')
    # Prediction (solid)
    plt.plot(time_ps[81:], pred_vals, linestyle='-', linewidth=4.5, color=color, label=legend_labels[i])
    # Ground truth (dotted + markers)
    plt.plot(time_ps[81:], true_vals, linewidth=0.0, color=color,
             marker='o', ms=10, markevery=10, alpha=0.9, label='_nolegend_')

plt.title("SB: CVNN", fontsize=45)
plt.xlabel(r"Time ($1/\Delta$)", fontsize=35)
plt.legend(fontsize=25, loc='upper right')
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=30)
#plt.tight_layout()
#plt.margins(x=0.0, y=0.0)

out_png = f"{sprefix}_combined_density_matrix_trajectory.png"
plt.savefig(out_png, dpi=300)
plt.show()
print("Saved plot:", out_png)

