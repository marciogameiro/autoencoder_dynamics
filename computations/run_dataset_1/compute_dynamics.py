from autoencoder_model import *
import CMGDB
import torch

model_weights_fname = 'ci_model_weights_1.pth'

# Load model weights from file
model.load_state_dict(torch.load(model_weights_fname, weights_only=True))

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

# Determine data bound box
x_min = min(np.concatenate((x_current_enc[:, 0], x_forward_enc[:, 0])))
x_max = max(np.concatenate((x_current_enc[:, 0], x_forward_enc[:, 0])))
y_min = min(np.concatenate((x_current_enc[:, 1], x_forward_enc[:, 1])))
y_max = max(np.concatenate((x_current_enc[:, 1], x_forward_enc[:, 1])))

delta = 0.20
dx, dy = x_max - x_min, y_max - y_min
x_min, x_max = x_min - delta * dx, x_max + delta * dx
y_min, y_max = y_min - delta * dy, y_max + delta * dy

@torch.no_grad()
def f_latent_map(x_input, model, device):
    x_tensor = torch.tensor(x_input, dtype=torch.float32).to(device)
    y = model.latent_map(x_tensor).cpu().numpy()
    return y

def f(x):
    x_input = [x]
    y = f_latent_map(x_input, model, device)
    return y[0]

def F(rect):
    return CMGDB.BoxMap(f, rect, padding=False)

subdiv_min = 14 # 16
subdiv_max = 28 # 24
subdiv_init = 10 # 10
subdiv_limit = 10000
lower_bounds = [x_min, y_min]
upper_bounds = [x_max, y_max]

dyn_model = CMGDB.Model(subdiv_min, subdiv_max, subdiv_init, subdiv_limit, lower_bounds, upper_bounds, F)

morse_graph, map_graph = CMGDB.ComputeConleyMorseGraph(dyn_model)

# Save Morse graph
fig_fname = 'ci_morse_graph'

gv_graph = CMGDB.PlotMorseGraph(morse_graph)

gv_graph.render(filename=fig_fname, format='pdf');

# Save Morse sets
fig_fname = 'ci_morse_sets.pdf'

CMGDB.PlotMorseSets(morse_graph, xlabel='$z_1$', ylabel='$z_2$', xlim=[lower_bounds[0], upper_bounds[0]],
                    ylim=[lower_bounds[1], upper_bounds[1]], fig_fname=fig_fname)
