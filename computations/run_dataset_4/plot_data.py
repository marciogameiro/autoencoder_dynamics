from autoencoder_model import *
import numpy as np
import torch

model_weights_fname = 'ci_model_weights_1.pth'

# Read training data from file
XY = np.loadtxt('train_data.csv', delimiter=',')

# Split each row into x and y vectors
XY_data = np.array([[np.array(xy[:N]), np.array(xy[N:])] for xy in XY])

# Move the training dataset to the selected device
XY_data = torch.tensor(np.array(XY_data), dtype=torch.float32).to(device)

# Load model weights from file
model.load_state_dict(torch.load(model_weights_fname, weights_only=True))

# Set to evaluation mode
model.eval();

with torch.no_grad():
    # Get the initial x states
    x_current = XY_data[:, 0, :]
    # Get the x states after one step
    x_forward = XY_data[:, 1, :]
    # Encode the initial and next states
    x_current_enc = model.encoder(x_current).cpu().numpy()
    x_forward_enc = model.encoder(x_forward).cpu().numpy()

x_enc_tracjectories = []
for k in range(num_trajectories):
    # Get x_encoded trajectory (all the x_current_enc points plus the last x_forward_enc point)
    x_enc_traj = np.vstack([x_current_enc[k * time_steps:(k + 1) * time_steps, :], x_forward_enc[time_steps, :]])
    x_enc_tracjectories.append(x_enc_traj)

x_enc_tracjectories = np.array(x_enc_tracjectories)

plt.figure(figsize=(12, 8))

colors = plt.cm.jet(np.linspace(0, 1, num_trajectories))

for k in range(num_trajectories):
    # Plot trajectories
    plt.plot(x_enc_tracjectories[k, :, 0], x_enc_tracjectories[k, :, 1], color=colors[k], linestyle='--', lw=1, alpha=0.8)
    plt.scatter(x_enc_tracjectories[k, :, 0], x_enc_tracjectories[k, :, 1], s=30, marker='o', edgecolors='k', zorder=20)
    # Plot starting point of trajectories
    plt.scatter(x_enc_tracjectories[k, 0, 0], x_enc_tracjectories[k, 0, 1], color='black', s=80, marker='*', zorder=10)

plt.xlabel('$z_1$')
plt.ylabel('$z_2$')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
