import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
import os
import math

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

ostl_steps = 80



### Custom Real Linear Layer
class RealLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(RealLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

        self.reset_parameters()

    # def reset_parameters(self, in_features):
    #    gain = init.calculate_gain('relu')
    #    scale = gain #/ (in_features) ** 0.5   # Adjusted for complex numbers
    #    init.xavier_uniform_(self.weight, gain=scale)
    #    init.zeros_(self.bias)  # Zero biases for stability
    def reset_parameters(self):
        # He initialization: variance = 2/n_in
        std = math.sqrt(2.0 / self.in_features)  # sqrt(2/n_in) for total variance
        bound = std * math.sqrt(3)  # Range: ±sqrt(6/n_in) for variance 2/n_in
        torch.nn.init.uniform_(self.weight, -bound, bound)
        # Bias initialization (small uniform range)
        # bound_bias = 1 / math.sqrt(self.in_features)
        # torch.nn.init.uniform_(self.bias, -bound_bias, bound_bias)
        # def reset_parameters(self) -> None:
        #    """Initialize weights using Kaiming initialization (for ReLU-based networks)."""
        #    init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='relu')
        #    if self.bias is not None:
        #        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        #        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        #        init.uniform_(self.bias, -bound, bound)
        init.zeros_(self.bias)

    def forward(self, input):
        return torch.matmul(input, self.weight.t()) + self.bias


class SpinBosonRealPINN(nn.Module):
    def __init__(self, sequence_length, state_size, hidden_size, ostl_steps):
        super(SpinBosonRealPINN, self).__init__()

        # Define input and output sizes
        input_size = sequence_length * state_size
        output_size = ostl_steps * state_size

        self.ostl_steps = ostl_steps
        self.state_size = state_size

        # Define the network layers
        self.fc1 = RealLinear(input_size, hidden_size)
        self.fc2 = RealLinear(hidden_size, hidden_size)
        self.fc3 = RealLinear(hidden_size, hidden_size)
        self.fc4 = RealLinear(hidden_size, hidden_size)
        self.fc5 = RealLinear(hidden_size, output_size)  # Final output

        self.activation = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten input

        # Forward pass
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.fc5(x)  # Output shape: (batch_size, 160)
        x = x.view(batch_size, self.ostl_steps, self.state_size)
        return x


### MSE Loss Function for Multi-Step Prediction
def mse_loss(pred, target):
    # pred and target shape: (batch_size, pred_steps, state_size)
    return nn.MSELoss()(pred, target)


def trace_penalty(pred):
    # Extract diagonal indices assuming standard real-valued ordering
    diag_indices = [0]
    idx = 0
    for i in range(1, 7):
        idx += (7 - i) * 2 + 1
        diag_indices.append(idx)

    trace = sum([pred[:, :, i] for i in diag_indices])
    return torch.mean((trace - 1) ** 2)


### Training Loop
def train_model(model, train_X, train_Y, val_X, val_Y, epochs, lr,lambda_trace, save_dir="saved_models"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)
    # Move model and data to GPU
    model.to(device)
    train_X, train_Y = train_X.to(device), train_Y.to(device)
    val_X, val_Y = val_X.to(device), val_Y.to(device)

    # Create directory to save models
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Directory {save_dir} is created successfully where the trained models will be saved")
    else:
        print(f"Directory {save_dir} already exists where the trained models will be saved")

    best_val_loss = float('inf')  # Track the best validation loss

    train_mse_loss = []
    train_trace_loss = []
    train_loss = []
    val_mse_loss = []
    val_trace_loss = []
    val_loss = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        pred = model(train_X)  # Shape: (batch_size, 40, 4)
        # Compute losses
        data_loss = mse_loss(pred, train_Y)
        penalty = trace_penalty(pred)
        total_loss = data_loss + lambda_trace * penalty
        # Append losses to lists
        train_mse_loss.append(data_loss.item())
        train_trace_loss.append(penalty.item())
        train_loss.append(total_loss.item())

        # Backpropagation
        total_loss.backward()
        optimizer.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_pred = model(val_X)
            val_data_loss = mse_loss(val_pred, val_Y)
            val_penalty = trace_penalty(val_pred)
            val_total_loss = val_data_loss + lambda_trace * val_penalty
            # Append losses to lists
            val_mse_loss.append(val_data_loss.item())
            val_trace_loss.append(val_penalty.item())
            val_loss.append(val_total_loss.item())

        # Print progress every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}: Train Loss = {total_loss.item():.4e}, Val Loss = {val_total_loss.item():.4e}')

        # Save the model if validation loss improves
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            model_filename = f"{save_dir}/real_mlp_model-{epoch + 1:02d}-tloss-{total_loss.item():.4e}-vloss-{val_total_loss.item():.4e}.pth"
            torch.save(model, model_filename)
            # print(f"Model saved at epoch {epoch+1} with validation loss {val_total_loss.item():.4e}")

    # Save all losses to a .npz file
    np.savez('fmo_rvnn_losses.npz',
             train_mse_loss=np.array(train_mse_loss),
             train_trace_loss=np.array(train_trace_loss),
             train_loss=np.array(train_loss),
             val_mse_loss=np.array(val_mse_loss),
             val_trace_loss=np.array(val_trace_loss),
             val_loss=np.array(val_loss))
    print(f"Losses saved to 'fmo_rvnn_losses.npz'")


def print_model_parameters(model):
    print("Layer-wise Parameter Count:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel()} parameters")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params}")


### Example Usage
if __name__ == "__main__":
    # Placeholder for data loading (replace with actual data)
    # Expected shapes:
    # train_X: (batch_size, 81, 4) real tensor (input sequence)
    # train_Y: (batch_size, 40, 4) real tensor (next 40 steps)
    # val_X: (batch_size, 81, 4) real tensor
    # val_Y: (batch_size, 40, 4) real tensor

    train_X = torch.from_numpy(np.load('train_X.npy')).to(torch.float32)
    train_Y = torch.from_numpy(np.load('train_Y.npy')).to(torch.float32)  # Multi-step targets
    val_X = torch.from_numpy(np.load('val_X.npy')).to(torch.float32)
    val_Y = torch.from_numpy(np.load('val_Y.npy')).to(torch.float32)  # Multi-step targets

    # Initialize the real-valued model for multi-step prediction
    model_real = SpinBosonRealPINN(sequence_length=81, state_size=49, hidden_size=128, ostl_steps=ostl_steps).to(device)

    # For a given model, e.g., real_model or complex_model:
    print("Total parameters:", print_model_parameters(model_real))

    # Train the model
    train_model(model_real, train_X, train_Y, val_X, val_Y, epochs=100000, lr=0.001, lambda_trace=1.0)
