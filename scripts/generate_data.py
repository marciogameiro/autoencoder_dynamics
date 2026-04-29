from autoencoder_model import *
import numpy as np

# Define the vector field and generate data
X, Y = generate_data_ci(num_trajectories, time_steps, tau, N)

# Save train data
data_fname = 'train_data.csv'

XY = np.concatenate((X, Y), axis=1)

np.savetxt(data_fname, XY, delimiter=',')
