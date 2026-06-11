from autoencoder_model import *

from scipy.optimize import fsolve
import numpy as np
import pickle

num_trajectories = 10000
time_steps = 60
tol = 1e-8

file_name = 'traj_attractors.pkl'

roots_fname = 'stable_solutions.csv'

# Compute stable steady state solutions

def steady_state_eq(a):
    """F(a) = 0 for finding fixed points"""
    return ci_vector_field(0, a)

# Stable Positive (near sqrt(alpha-1))
guess_pos = np.zeros(N); guess_pos[0] = np.sqrt(alpha - 1.0)
root_pos = fsolve(steady_state_eq, guess_pos)

# Stable Negative
guess_neg = np.zeros(N); guess_neg[0] = -np.sqrt(alpha - 1.0)
root_neg = fsolve(steady_state_eq, guess_neg)

stable_roots = np.array([root_neg.T, root_pos.T])

# Save stable steady state solutions
np.savetxt(roots_fname, stable_roots, delimiter=',')

# Compute trajactories to test convergence to attractor

# Define the vector field and generate data
X, Y = generate_data_ci(num_trajectories, time_steps, tau, N)

traj_attractors = {}
for k in range(num_trajectories):
	# Get initial and last point of k-th trajectory
	x = X[k * time_steps, :]
	y = Y[(k + 1) * time_steps - 1, :]
	# Compare the first 16 entries for convergence
	att = 0 # Did not converge to an attractor
	if np.linalg.norm(y[:16] - root_neg[:16]) <= tol:
		att = -1
	if np.linalg.norm(y[:16] - root_pos[:16]) <= tol:
		att = 1
	traj_attractors[tuple(x)] = att

with open(file_name, 'wb') as file:
    pickle.dump(traj_attractors, file)
