import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
import os
import math

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

ostl_steps = 40
n_states = 2

### Custom Complex Linear Layer
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Initialize real and imaginary weights and biases
        self.weight_real = nn.Parameter(torch.randn(out_features, in_features))
        self.weight_imag = nn.Parameter(torch.randn(out_features, in_features))
        self.bias_real = nn.Parameter(torch.randn(out_features))
        self.bias_imag = nn.Parameter(torch.randn(out_features))
        #self.init_weights(in_features)
        self.reset_parameters()

    def reset_parameters(self):
        # He initialization: variance = 1/n_in per part, total 2/n_in
        std = math.sqrt(1.0 / self.in_features)  # sqrt(1/n_in) for each part
        torch.nn.init.uniform_(self.weight_real, -std * math.sqrt(3), std * math.sqrt(3))
        torch.nn.init.uniform_(self.weight_imag, -std * math.sqrt(3), std * math.sqrt(3))
        # Bias initialization (optional, small uniform range)
        #bound = 1 / math.sqrt(self.in_features)
        #torch.nn.init.uniform_(self.bias_real, -bound, bound)
        #torch.nn.init.uniform_(self.bias_imag, -bound, bound)


        init.zeros_(self.bias_real)  # Zero biases for stability
        init.zeros_(self.bias_imag)

    def forward(self, input):
        # Separate real and imaginary parts of the input
        input_real = input.real
        input_imag = input.imag
        # Complex matrix multiplication
        output_real = (torch.matmul(input_real, self.weight_real.t()) - 
                       torch.matmul(input_imag, self.weight_imag.t()) + 
                       self.bias_real)
        output_imag = (torch.matmul(input_real, self.weight_imag.t()) + 
                       torch.matmul(input_imag, self.weight_real.t()) + 
                       self.bias_imag)
        return torch.complex(output_real, output_imag)

### Custom Complex GELU Activation
### Custom ModReLU Activation (Phase-Preserving)
class ModReLU(nn.Module):
    def __init__(self, hidden_size):
        super(ModReLU, self).__init__()
        self.bias = nn.Parameter(torch.zeros(hidden_size))  # Learnable bias per neuron
    
    def forward(self, x):
        magnitude = torch.abs(x)
        threshold = torch.relu(magnitude - self.bias.unsqueeze(0))
        phase = torch.angle(x)
        return threshold * torch.exp(1j * phase)

class CELU(nn.Module):
    def __init__(self):
        super(CELU, self).__init__()
        self.elu = nn.ELU()  # Standard ELU activation

    def forward(self, z):
        real_part = self.elu(z.real)
        imag_part = self.elu(z.imag)
        return torch.complex(real_part, imag_part)

class CSig(nn.Module):
    def __init__(self):
        super(CSig, self).__init__()
        self.sig = nn.Sigmoid()  # Standard ELU activation

    def forward(self, z):
        real_part = self.sig(z.real)
        imag_part = self.sig(z.imag)
        return torch.complex(real_part, imag_part)



class CGELU(nn.Module):
    def __init__(self):
        super(CGELU, self).__init__()
        self.gelu = nn.GELU()  # Standard ELU activation

    def forward(self, z):
        real_part = self.gelu(z.real)
        imag_part = self.gelu(z.imag)
        return torch.complex(real_part, imag_part)

class CRELU(nn.Module):
    def __init__(self):
        super(CRELU, self).__init__()
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Apply GELU separately to real and imaginary parts
        real_part = self.relu(x.real)
        imag_part = self.relu(x.imag)
        return torch.complex(real_part, imag_part)

class CLRELU(nn.Module):
    def __init__(self):
        super(CLRELU, self).__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=math.sqrt(5))
    
    def forward(self, x):
        # Apply GELU separately to real and imaginary parts
        real_part = self.lrelu(x.real)
        imag_part = self.lrelu(x.imag)
        return torch.complex(real_part, imag_part)

### CV-PINN Model for Spin-Boson Data
class SpinBosonCVPINN(nn.Module):
    def __init__(self, sequence_length, state_size, hidden_size, ostl_steps):
        super(SpinBosonCVPINN, self).__init__()
        self.ostl_steps = ostl_steps
        self.state_size = state_size

        input_size = sequence_length * state_size
        output_size = ostl_steps * state_size
        # Define the network layers
        self.fc1 = ComplexLinear(input_size, hidden_size)
        self.fc2 = ComplexLinear(hidden_size, hidden_size)
        self.fc3 = ComplexLinear(hidden_size, hidden_size)
        self.fc4 = ComplexLinear(hidden_size, hidden_size)
        self.fc5 = ComplexLinear(hidden_size, output_size)  # Output: 3 complex numbers (rho_11, rho_12, rho_22)
        self.activation = CRELU()
    
    def forward(self, x):
        batch_size = x.size(0)
        # Flatten input to (batch_size, input_size)
        x = x.reshape(batch_size, -1)  # Use reshape instead of view
        # Forward pass through the network
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.fc5(x)  # Output shape: (batch_size, 3), complex

        x = x.view(batch_size, self.ostl_steps, self.state_size)
        return x

### Complex MSE Loss Function
def complex_mse_loss(pred, target):
    pred = pred.view(-1, pred.shape[-1])
    target = target.view(-1, target.shape[-1])

    pred_real = pred.real
    pred_imag = pred.imag
    target_real = target.real
    target_imag = target.imag
    # Compute MSE for real and imaginary parts
    mse_real = nn.MSELoss()(pred_real, target_real)
    mse_imag = nn.MSELoss()(pred_imag, target_imag)
    # Average the two losses
    return (mse_real + mse_imag) / 2

# Trace Penalty Loss Function
def trace_penalty(pred):
    # pred shape: (batch_size, pred_steps, state_size)
    rho_11 = pred[:, :, 0]
    rho_22 = pred[:, :, 2]
    #Trace condition: real(rho_11) + real(rho_22) = 1 for each time step
    trace = rho_11.real + rho_22.real  # Shape: (batch_size, pred_steps)
    # Compute penalty for each time step and average
    penalty = torch.mean((trace - 1) ** 2)
    return penalty  # Penalty for deviation from 1

# Training Loop
def train_model(model, train_X, train_Y, val_X, val_Y, epochs, lr, lambda_trace, save_dir="saved_models"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)
    model.to(device)
    train_X, train_Y = train_X.to(device), train_Y.to(device)
    val_X, val_Y = val_X.to(device), val_Y.to(device)
    # Create directory to save models
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print("Directory", save_dir, "is created successfully where the trained models will be saved")
    else:
        print("Directory", save_dir, "already exists where the trained models will be saved")

    best_val_loss = float('inf')  # Initialize best validation loss as infinity

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
        pred = model(train_X)
        # Compute losses
        data_loss = complex_mse_loss(pred, train_Y)
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
            val_data_loss = complex_mse_loss(val_pred, val_Y)
            val_penalty = trace_penalty(val_pred)
            val_total_loss = val_data_loss + lambda_trace * val_penalty
            
            # Append losses to lists
            val_mse_loss.append(val_data_loss.item())
            val_trace_loss.append(val_penalty.item())
            val_loss.append(val_total_loss.item())

        
        # Print progress every 100 epochs
        if (epoch + 1) % 1000 == 0:
            print(f'Epoch {epoch+1}: Train Loss = {total_loss.item():.4e}, Val Loss = {val_total_loss.item():.4e}')

        # Save the model if validation loss improves
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            model_filename = f"{save_dir}/cv_mlp_model-{epoch+1:02d}-tloss-{total_loss.item():.4e}-vloss-{val_total_loss.item():.4e}.pth"
            torch.save(model, model_filename)
            print(f"Model saved at epoch {epoch+1} with validation loss {val_total_loss.item():.4e}")

    # Save all losses to a .npz file
    np.savez('sbcvnn_losses.npz',
             train_mse_loss=np.array(train_mse_loss),
             train_trace_loss=np.array(train_trace_loss),
             train_loss=np.array(train_loss),
             val_mse_loss=np.array(val_mse_loss),
             val_trace_loss=np.array(val_trace_loss),
             val_loss=np.array(val_loss))
    print(f"Losses saved to 'sbcvnn_losses.npz'")

def print_model_parameters(model):
    print("Layer-wise Parameter Count:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel()} parameters")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params}")


def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)

### Example Usage
if __name__ == "__main__":
    # Placeholder for data loading (replace with your actual data)
    # Expected shapes:
    # train_X: (batch_size, 81, 3) complex tensor
    # train_Y: (batch_size, 3) complex tensor
    # val_X: (batch_size, 81, 3) complex tensor
    # val_Y: (batch_size, 3) complex tensor
    """
    train_X = torch.from_numpy(np.load('train_X_complex.npy'), dtype=torch.complex64)
    train_Y = torch.from_numpy(np.load('train_Y_complex.npy'), dtype=torch.complex64)
    val_X = torch.from_numpy(np.load('val_X_complex.npy'), dtype=torch.complex64)
    val_Y = torch.from_numpy(np.load('val_Y_complex.npy'), dtype=torch.complex64)
    """
    # Load the saved NumPy files
    train_X = np.load('train_X.npy')
    train_Y = np.load('train_Y.npy')
    val_X = np.load('val_X.npy')
    val_Y = np.load('val_Y.npy')

    # Convert to PyTorch tensors with complex64 type
    train_X = torch.from_numpy(train_X).to(torch.complex64)
    train_Y = torch.from_numpy(train_Y).to(torch.complex64)
    val_X = torch.from_numpy(val_X).to(torch.complex64)
    val_Y = torch.from_numpy(val_Y).to(torch.complex64)
    # Initialize the model
    model = SpinBosonCVPINN(sequence_length=81, state_size=3, hidden_size=42, ostl_steps = ostl_steps)
    
    # For a given model, e.g., real_model or complex_model:
    print("Total parameters:", print_model_parameters(model))


    # Uncomment and replace with your data to train
    train_model(model, train_X, train_Y, val_X, val_Y, epochs=100000, lr=0.001, lambda_trace=1.0)
