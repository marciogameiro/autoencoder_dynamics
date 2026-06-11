import CMGDB
from autoencoder_model import *
from basins_attraction import *

import matplotlib.pyplot as plt
import torch
import pickle

file_name = 'traj_attractors.pkl'

roots_fname = 'stable_solutions.csv'

model_weights_fname = 'ci_model_weights.pth'

with open(file_name, 'rb') as file:
    traj_attractors = pickle.load(file)

# Get initial points data
X = np.array([list(x) for x in traj_attractors])

# Move dataset to the selected device
X = torch.tensor(np.array(X), dtype=torch.float32).to(device)

# Read stable steady state solutions data
stable_roots = np.loadtxt(roots_fname, delimiter=',')

# Move data to the selected device
stable_roots = torch.tensor(np.array(stable_roots), dtype=torch.float32).to(device)

# Read training data from file
XY = np.loadtxt('train_data.csv', delimiter=',')

# Split each row into x and y vectors
XY_data = np.array([[np.array(xy[:N]), np.array(xy[N:])] for xy in XY])

# Move the training dataset to the selected device
XY_data = torch.tensor(np.array(XY_data), dtype=torch.float32).to(device)

# Load model weights from file
model.load_state_dict(torch.load(model_weights_fname, weights_only=True))

# Set to evaluation mode
model.eval()

with torch.no_grad():
    # Encode the initial points
    X_enc = model.encoder(X).cpu().numpy()
    # Encode stable steady states
    stable_roots_enc = model.encoder(stable_roots).cpu().numpy()

with torch.no_grad():
    # Get the initial x states
    x_current = XY_data[:, 0, :]
    # Get the x states after one step
    x_forward = XY_data[:, 1, :]
    # Encode the initial and next states
    x_current_enc = model.encoder(x_current).cpu().numpy()
    x_forward_enc = model.encoder(x_forward).cpu().numpy()

# Determine data bound box
x_min = min(np.concatenate((x_current_enc[:, 0], x_forward_enc[:, 0])))
x_max = max(np.concatenate((x_current_enc[:, 0], x_forward_enc[:, 0])))
y_min = min(np.concatenate((x_current_enc[:, 1], x_forward_enc[:, 1])))
y_max = max(np.concatenate((x_current_enc[:, 1], x_forward_enc[:, 1])))

delta = 0.1
dx, dy = x_max - x_min, y_max - y_min
x_min, x_max = x_min - delta * dx, x_max + delta * dx
y_min, y_max = y_min - delta * dy, y_max + delta * dy

subdiv_min = 16
subdiv_max = 16
subdiv_init = 16
subdiv_limit = 10000
lower_bounds = [x_min, y_min]
upper_bounds = [x_max, y_max]

if subdiv_max % 2 == 0:
    n1 = n2 = subdiv_max // 2
else:
    n1 = (subdiv_max + 1) // 2
    n2 = n1 - 1

@torch.no_grad()
def f_latent_map(x_input, model, device):
    x_tensor = torch.tensor(x_input, dtype=torch.float32).to(device)
    y = model.latent_map(x_tensor).cpu().numpy()
    return y

x = np.linspace(lower_bounds[0], upper_bounds[0], 2**n1 + 1)
y = np.linspace(lower_bounds[1], upper_bounds[1], 2**n2 + 1)
X, Y = np.meshgrid(x, y)

X_pts = np.vstack([X.ravel(), Y.ravel()]).T

Y_pts = f_latent_map(X_pts, model, device)

def f(x):
    # Find point in the grid closest to x
    matching_rows = np.all(np.isclose(X_pts, x, rtol=1e-12, atol=1e-12), axis=1)
    indices = np.where(matching_rows)[0]
    return Y_pts[indices[0]]

def F(rect):
    return CMGDB.BoxMap(f, rect, padding=True)

dyn_model = CMGDB.Model(subdiv_min, subdiv_max, subdiv_init, subdiv_limit, lower_bounds, upper_bounds, F)

morse_graph, map_graph = CMGDB.ComputeConleyMorseGraph(dyn_model)

# # Save Morse graph
# fig_fname = 'ci_morse_graph_uniform'

# gv_graph = CMGDB.PlotMorseGraph(morse_graph)

# gv_graph.render(filename=fig_fname, format='pdf');

# # Save Morse sets
# fig_fname = 'ci_morse_sets_uniform.pdf'

# CMGDB.PlotMorseSets(morse_graph, xlabel='$z_1$', ylabel='$z_2$', xlim=[lower_bounds[0], upper_bounds[0]],
#                     ylim=[lower_bounds[1], upper_bounds[1]], fig_fname=fig_fname)

# Compute basins of attraction for minimal attractors
attractors = [k for k in range(morse_graph.num_vertices()) if len(list(morse_graph.adjacencies(k))) == 0]
att_basins = attractor_basins(map_graph, morse_graph, attractors)

def in_attractor_basin(z, attractor_basin, morse_graph):
    """Check if point z is in a given collection of boxes (attractor_basin)"""
    for b in attractor_basin:
        x_lo, y_lo, x_hi, y_hi = morse_graph.phase_space_box(b)
        if x_lo <= z[0] <= x_hi and y_lo <= z[1] <= y_hi:
            return True
    return False

# Encoded positive and negative roots
neg_root_enc = stable_roots_enc[0, :]
pos_root_enc = stable_roots_enc[1, :]

# Assuming there are two minimal attractors
assert len(attractors) == 2

att_0, att_1 = attractors[0], attractors[1]

if in_attractor_basin(neg_root_enc, att_basins[att_0], morse_graph):
    att_neg_root = att_0
elif in_attractor_basin(neg_root_enc, att_basins[att_1], morse_graph):
    att_neg_root = att_1
else:
    raise ValueError('Negative root is not in an attractor basin.')

if in_attractor_basin(pos_root_enc, att_basins[att_0], morse_graph):
    att_pos_root = att_0
elif in_attractor_basin(pos_root_enc, att_basins[att_1], morse_graph):
    att_pos_root = att_1
else:
    raise ValueError('Positive root is not in an attractor basin.')

# Basins for negative and positive roots
att_basins_neg = att_basins[att_neg_root]
att_basins_pos = att_basins[att_pos_root]

# Compute statistics of points encoded to attractor basins
num_not_attractor = 0
num_not_classified = 0
num_misclassified_neg = 0
num_misclassified_pos = 0
num_correct_neg = 0
num_correct_pos = 0
for k, (x, root_sign) in enumerate(traj_attractors.items()):
    if root_sign == 0:
        num_not_attractor += 1
        continue
    if in_attractor_basin(X_enc[k, :], att_basins_neg, morse_graph):
        if root_sign == -1:
            num_correct_neg += 1
        else:
            num_misclassified_neg += 1
        continue
    if in_attractor_basin(X_enc[k, :], att_basins_pos, morse_graph):
        if root_sign == 1:
            num_correct_pos += 1
        else:
            num_misclassified_pos += 1
        continue
    num_not_classified += 1

print('Total number of attractor trajectories:', len(traj_attractors))
print('Number of non-converging trajectories:', num_not_attractor)
print('Number of points not classified (not in an attractor basin):', num_not_classified)
print('Number of points misclassified in basin of negative root:', num_misclassified_neg)
print('Number of points misclassified in basin of positive root:', num_misclassified_pos)
print('Number of points correctly classified in basin of negative root:', num_correct_neg)
print('Number of points correctly classified in basin of positive root:', num_correct_pos)

# Plot minimal attractors and basins
clist = ['#1f77b4', '#e6550d']

basin_alpha = 0.35
morse_alpha = 1.0

fig_w, fig_h = 8, 8

fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)

# Draw basins of attraction first
for k, att in enumerate(attractors):
    clr = clist[k]
    plot_scatter_boxes(att_basins[att], clr, basin_alpha, morse_graph, lower_bounds, upper_bounds, fig, ax)
    plot_scatter_boxes(morse_graph.morse_set(att), clr, morse_alpha, morse_graph, lower_bounds, upper_bounds, fig, ax)

# Plot encoded initial points
# ax.scatter(X_enc[:, 0], X_enc[:, 1], s=30, marker='o', edgecolors='k')
# ax.scatter(stable_roots_enc[:, 0], stable_roots_enc[:, 1], s=30, marker='o', color='r', edgecolors='k')

x_min_plot, x_max_plot = lower_bounds[0], upper_bounds[0]
y_min_plot, y_max_plot = lower_bounds[1], upper_bounds[1]
ax.set_xlim([x_min_plot, x_max_plot])
ax.set_ylim([y_min_plot, y_max_plot])

ax.set_xlabel('$z_1$', fontsize=15)
ax.set_ylabel('$z_2$', fontsize=15)
ax.tick_params(labelsize=13)

fig.savefig('ci_attractor_basins.pdf', bbox_inches='tight')
plt.show()
