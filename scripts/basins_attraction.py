import networkx as nx

def reachable_morse_sets(digraph_nx, morse_sets_by_idx):
    """Compute the Morse sets reachable from each box in the grid"""
    # Compute SCCs and condensation graph
    sccs = list(nx.strongly_connected_components(digraph_nx))
    condensation = nx.condensation(digraph_nx, sccs)
    # Map each box to its SCC index (same as condensation node)
    # The SCC index is the same as the condensation node
    box_to_scc_idx = {}
    for k, scc in enumerate(sccs):
        for b in scc:
            box_to_scc_idx[b] = k
    # Map boxes in Morse sets to Morse node
    box_to_morse_node = {}
    for k in morse_sets_by_idx:
        for b in morse_sets_by_idx[k]:
            box_to_morse_node[b] = k
    # Mapp SCC indices to Morse nodes
    scc_node_to_morse_node = {}
    for k, scc in enumerate(sccs):
        for b in scc:
            if b in box_to_morse_node:
                scc_node_to_morse_node[k] = box_to_morse_node[b]
                break
    # Traverse the reversed topologically sorted condensation graph
    # to find Morse sets reachable from each node in the graph
    reachable = {}
    for c in reversed(list(nx.topological_sort(condensation))):
        reach_c = set()
        # Find Morse nodes reachable from c
        if c in scc_node_to_morse_node:
            reach_c.add(scc_node_to_morse_node[c])
        for succ in condensation.successors(c):
            reach_c |= reachable[succ]
        reachable[c] = frozenset(reach_c)
    # Map each box to its reachable Morse set indices
    reachable_morse_nodes = {b: reachable[box_to_scc_idx[b]] for b in digraph_nx.nodes()}
    return reachable_morse_nodes

def attractor_basins(map_graph, morse_graph, attractors=None):
    # Get MVM and Morse graph sizes
    num_boxes = map_graph.num_vertices()
    num_morse_nodes  = morse_graph.num_vertices()
    # Make networkx digraph for MVM
    digraph_nx = nx.DiGraph()
    digraph_nx.add_nodes_from(range(num_boxes))
    for u in range(num_boxes):
        for v in map_graph.adjacencies(u):
            digraph_nx.add_edge(u, v)
    # Get a mapping from Morse node index to Morse sets
    morse_sets_by_idx = {k: frozenset(morse_graph.morse_set(k)) for k in range(num_morse_nodes)}
    if attractors == None:
        # Compute basins for all attractors (minimal nodes)
        attractors = [k for k in range(num_morse_nodes) if len(list(morse_graph.adjacencies(k))) == 0]
    # Get list of reachable Morse set from each box
    reachable = reachable_morse_sets(digraph_nx, morse_sets_by_idx)
    # Get basin of attraction for each attractor in attractors
    att_basins = {}
    for att in attractors:
        # Get set of boxes that reach the single Morse set att
        basin = sorted(b for b, reach in reachable.items() if reach == {att})
        att_basins[att] = basin
    return att_basins

def plot_scatter_boxes(box_indices, color, alpha, morse_graph, lower_bounds, upper_bounds, fig, ax):
    """Scatter plot a set of boxes as square markers sized to match each box."""
    x_min_plot, x_max_plot = lower_bounds[0], upper_bounds[0]
    y_min_plot, y_max_plot = lower_bounds[1], upper_bounds[1]
    x_axis_width  = x_max_plot - x_min_plot
    y_axis_height = y_max_plot - y_min_plot
    s0_x = (ax.get_window_extent().width  / x_axis_width)  * (72.0 / fig.dpi)
    s0_y = (ax.get_window_extent().height / y_axis_height) * (72.0 / fig.dpi)
    X, Y, S = [], [], []
    for b in box_indices:
        rect = morse_graph.phase_space_box(b)   # [x_lo, y_lo, x_hi, y_hi]
        dim  = len(rect) // 2
        cx   = (rect[0]     + rect[dim])     / 2
        cy   = (rect[1]     + rect[dim + 1]) / 2
        w    =  rect[dim]   - rect[0]
        h    =  rect[dim+1] - rect[1]
        X.append(cx)
        Y.append(cy)
        S.append(max((s0_x * w) ** 2, (s0_y * h) ** 2))
    if X:
        ax.scatter(X, Y, s=S, marker='s', c=color, alpha=alpha, linewidths=0)
