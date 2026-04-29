from autoencoder_model import *
import CMGDB
import torch

# Load model weights from file
model.load_state_dict(torch.load('ci_model_weights.pth', weights_only=True))

# Set to evaluation mode
model.eval();

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
lower_bounds = [-3.0, -2.0]
upper_bounds = [3.0, 2.0]

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
