import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
import os
import math
import torch.optim as optim

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

ostl_steps = 80
sequence_length = 81
state_size = 36  # Adjusted for FMO complex (8 sites)

### Custom Complex Linear Layer
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.weight_real = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_imag = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_real = nn.Parameter(torch.zeros(out_features))
        self.bias_imag = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(1.0 / self.weight_real.shape[1])
        torch.nn.init.uniform_(self.weight_real, -std * math.sqrt(3), std * math.sqrt(3))
        torch.nn.init.uniform_(self.weight_imag, -std * math.sqrt(3), std * math.sqrt(3))

    def forward(self, input):
        input_real = input.real
        input_imag = input.imag
        output_real = torch.matmul(input_real, self.weight_real.t()) - torch.matmul(input_imag, self.weight_imag.t()) + self.bias_real
        output_imag = torch.matmul(input_real, self.weight_imag.t()) + torch.matmul(input_imag, self.weight_real.t()) + self.bias_imag
        return torch.complex(output_real, output_imag)

class CRELU(nn.Module):
    def __init__(self):
        super(CRELU, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return torch.complex(self.relu(x.real), self.relu(x.imag))

### CV-PINN Model for FMO Complex
class SpinBosonCVPINN(nn.Module):
    def __init__(self, sequence_length, state_size, hidden_size, ostl_steps):
        super(SpinBosonCVPINN, self).__init__()
        input_size = sequence_length * state_size
        output_size = ostl_steps * state_size
        self.fc1 = ComplexLinear(input_size, hidden_size)
        self.fc2 = ComplexLinear(hidden_size, hidden_size)
        self.fc3 = ComplexLinear(hidden_size, hidden_size)
        self.fc4 = ComplexLinear(hidden_size, hidden_size)
        self.fc5 = ComplexLinear(hidden_size, output_size)
        self.activation = CRELU()
        self.ostl_steps = ostl_steps
        self.state_size = state_size

    def forward(self, x):
        b = x.size(0)
        x = x.reshape(b, -1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.fc5(x)
        return x.view(b, self.ostl_steps, self.state_size)

### Complex MSE Loss Function
def complex_mse_loss(pred, target):
    pred = pred.view(-1, pred.shape[-1])
    target = target.view(-1, target.shape[-1])
    mse_real = nn.MSELoss()(pred.real, target.real)
    mse_imag = nn.MSELoss()(pred.imag, target.imag)
    return (mse_real + mse_imag) / 2


def trace_penalty(pred):
    """
    Compute trace penalty for pred tensor of shape (batch_size, num_matrices, 36),
    where 36 elements correspond to upper-triangular entries (including diagonal)
    of a 8x8 matrix in row-major order.

    The penalty encourages the trace (sum of diagonal elements) to be close to 1.

    Returns a scalar tensor (penalty).
    """

    # Correct diagonal indices for upper-triangular vector of size 28 from 7x7 matrix
    diag_indices = [0, 8, 15, 21, 26, 30, 33, 35]

    # Extract diagonal elements (real part)
    diag_elements = pred[:, :, diag_indices].real  # shape: (batch_size, num_matrices, 7)

    # Sum diagonal elements to get trace for each matrix
    trace = diag_elements.sum(dim=-1)  # shape: (batch_size, num_matrices)

    # Compute penalty as mean squared deviation from 1
    penalty = torch.mean((trace - 1) ** 2)

    return penalty


saved_models = []


def train_model(model, train_X, train_Y, val_X, val_Y, epochs, lambda_trace, lr, save_dir="saved_models"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)
    model.to(device)
    train_X, train_Y = train_X.to(device), train_Y.to(device)
    val_X, val_Y = val_X.to(device), val_Y.to(device)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print("Directory", save_dir, "is created successfully")

    train_mse_loss = []
    train_trace_loss = []
    train_loss = []
    val_mse_loss = []
    val_trace_loss = []
    val_loss = []

    saved_models = []  # list of (val_loss, filepath)


    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(train_X)
        data_loss = complex_mse_loss(pred, train_Y)
        penalty = trace_penalty(pred)
        total_loss = data_loss + lambda_trace * penalty

        # Save training loss values
        train_mse_loss.append(data_loss.item())
        train_trace_loss.append(penalty.item())
        train_loss.append(total_loss.item())

        total_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(val_X)
            val_data_loss = complex_mse_loss(val_pred, val_Y)
            val_penalty = trace_penalty(val_pred)
            val_total_loss = val_data_loss + lambda_trace * val_penalty

            # Save validation loss values
            val_mse_loss.append(val_data_loss.item())
            val_trace_loss.append(val_penalty.item())
            val_loss.append(val_total_loss.item())

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}: Train Loss = {total_loss.item():.4e}, Val Loss = {val_total_loss.item():.4e}")

        # Save model and keep only best 5
        model_filename = f"{save_dir}/fmo_model_epoch{epoch+1:04d}_vloss{val_total_loss.item():.4e}.pth"
        torch.save(model, model_filename)
        saved_models.append((val_total_loss.item(), model_filename))
        saved_models.sort(key=lambda x: x[0])  # sort by loss

        if len(saved_models) > 5:
            _, file_to_remove = saved_models.pop()  # remove worst
            if os.path.exists(file_to_remove):
                os.remove(file_to_remove)
                print(f"Deleted older model: {file_to_remove}")

    # Save all training and validation losses
    np.savez('fmo8site_cvnn_losses.npz',
             train_mse_loss=np.array(train_mse_loss),
             train_trace_loss=np.array(train_trace_loss),
             train_loss=np.array(train_loss),
             val_mse_loss=np.array(val_mse_loss),
             val_trace_loss=np.array(val_trace_loss),
             val_loss=np.array(val_loss))
    print("Loss history saved to 'fmo8site_cvnn_losses.npz'")

def print_model_parameters(model):
    print("Layer-wise Parameter Count:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel()} parameters")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params}")

def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)

### Run Example
if __name__ == "__main__":
    train_X = torch.from_numpy(np.load('train_X.npy')).to(torch.complex64)
    train_Y = torch.from_numpy(np.load('train_Y.npy')).to(torch.complex64)
    val_X = torch.from_numpy(np.load('val_X.npy')).to(torch.complex64)
    val_Y = torch.from_numpy(np.load('val_Y.npy')).to(torch.complex64)

    print("train_X shape:", train_X.shape)
    print("train_Y shape:", train_Y.shape)

    model = SpinBosonCVPINN(sequence_length=sequence_length, state_size=state_size, hidden_size=111, ostl_steps=ostl_steps)
    print("Model has", sum(p.numel() for p in model.parameters() if p.requires_grad), "trainable parameters.")

    train_model(model, train_X, train_Y, val_X, val_Y, epochs=100000, lr=0.001, lambda_trace=1.0)
