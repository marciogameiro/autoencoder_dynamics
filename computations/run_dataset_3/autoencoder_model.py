import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import numpy as np

N = 64

alpha = 28.0
tau = 0.1
num_trajectories = 1000
time_steps = 30

# ==========================================
# Define the vector field
# ==========================================
L_eig = -np.arange(1, N + 1)**2

def compute_nonlinear(a):
    a_ext = np.concatenate((-a[::-1], [0], a))
    conv2 = np.convolve(a_ext, a_ext)
    conv3 = np.convolve(conv2, a_ext)
    center_index = 3 * N
    return conv3[center_index + 1 : center_index + 1 + N]

def f_chafee_infante(a, alpha):
    return (alpha + L_eig) * a + (alpha / 4.0) * compute_nonlinear(a)

def ci_vector_field(t, a):
    return f_chafee_infante(a, alpha)

# ==========================================
# Generate data
# ==========================================
def generate_data_ci(num_trajectories, time_steps, tau, N):
    """Generate data for the Chafee-Infante equation"""
    # Random seed for reproducibility
    # rand_seed = 7206
    # rand_seed = 42
    rand_seed = np.random.randint(10000)
    print('Random seed for initial conditions:', rand_seed)
    np.random.seed(rand_seed)

    # Create data sets X and Y from time series data, such
    # that phi_tau(x) = y for x = X[k, :] and y = Y[k, :],
    # where phi_tau is the ODE time-tau map
    X, Y = [], []
    t_span = [0, tau * time_steps]
    t_eval = np.linspace(t_span[0], t_span[1], time_steps + 1)

    for _ in range(num_trajectories):
        # Solve ODE with a random initial condition
        a0 = np.random.uniform(-2.0, 2.0, N) * np.exp(-0.5 * np.arange(N))
        sol = solve_ivp(ci_vector_field, t_span, a0, t_eval=t_eval, method='RK45')
        # Append data to X and Y
        traj = sol.y.T      # Shape: (time_steps + 1, N)
        X.append(traj[:-1]) # Current states
        Y.append(traj[1:])  # Next states
    return np.vstack(X), np.vstack(Y)

# Check for NVIDIA GPU (CUDA)
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using NVIDIA GPU')
# Use standard CPU
else:
    device = torch.device('cpu')
    print('Using standard CPU')

learning_rate = 0.003
num_epochs = 4000

# Define the nueral network
class DynamicsAutoencoder(nn.Module):
    def __init__(self, input_dim=64, latent_dim=2):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(), # nn.ReLU(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, latent_dim)
        )

        self.latent_map = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.Tanh(), # nn.ReLU(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, input_dim)
        )

# Send the model to the device
model = DynamicsAutoencoder(input_dim=N, latent_dim=2).to(device)
# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Attach the scheduler to the optimizer
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=100,
    min_lr=1e-6,
)

criterion = nn.MSELoss()
