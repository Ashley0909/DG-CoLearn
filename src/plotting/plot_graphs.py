import re
import matplotlib.pyplot as plt
import graphviz
import os
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
import time
import torch
from torch_geometric.utils import to_undirected
from collections import defaultdict

def extract_accuracy_from_file(file_path):
    accuracies = []
    with open(file_path, 'r') as file:
        for line in file:
            # Search for the line that contains the accuracy
            match = re.search(r'accuracy\s*=\s*([0-9.]+)', line)
            if match:
                accuracy = float(match.group(1))  # Extract the accuracy value
                accuracies.append(accuracy)
    return accuracies

def extract_loss_from_file(file_path, search_phrase):
    losses = []
    with open(file_path, 'r') as file:
        for line in file:
            # Search for the line that contains the loss
            match = re.search(search_phrase, line)
            if match:
                loss = float(match.group(1))
                losses.append(format(loss, '.17g')) # Extract the loss value in 17 significant figures
                # losses.append(loss)
    return losses

def draw_graph(edge_index, name, client=None):
    if client is None:
        path = 'graph_output/global'+name
    else:
        path = 'graph_output/'+name+'client'+str(client)

    # Create a graph
    dot = graphviz.Graph()

    # Make edge index undirected
    undirected_edge_index = to_undirected(edge_index)

    # Add nodes and edges from tensor
    for i in range(undirected_edge_index.shape[1]):
        dot.edge(str(undirected_edge_index[0, i].item()), str(undirected_edge_index[1, i].item()))

    # Render the graph
    dot.render(path, format='png')

def draw_adj_list(adj_list, subnodes=None, name="global"):
    path = 'graph_output/'+name

    dot = graphviz.Graph(format='png')

    if subnodes is None:
        subnodes = np.arange(len(adj_list))

    # Add edges from adjacency list
    for src in subnodes:
        for dst in adj_list[src]:
            if dst in subnodes:
                dot.edge(str(src), str(dst))

    dot.render(path, format='png')

def colour_adj_list(adj_list, labelling):
    path = 'graph_output/coloured'
    dot = graphviz.Graph(format='png')        

    # color_palette = ["green", "yellow", "red", "blue", "purple", "orange", "pink", "cyan", "brown", "gray"]
    color_palette = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
    "#5254a3", "#637939", "#e7ba52", "#d6616b", "#ce6dbd"
    ]

    unique_labels = list(set(labelling))
    color_map = {label: color_palette[i % len(color_palette)] for i, label in enumerate(unique_labels)}  # Assigns different colors

    # Add nodes with colors
    if isinstance(labelling, defaultdict):
        for label, nodes in labelling.items():
            for node in nodes:
                if adj_list[node] != []:
                    dot.node(str(node), style="filled", fillcolor=color_map[label])
    else:
        for node, label in enumerate(labelling):
            if adj_list[node] != []:
                dot.node(str(node), style="filled", fillcolor=color_map[label])

    # Add edges from adjacency list
    for src , neigh in enumerate(adj_list):
        for dst in neigh:
            dot.edge(str(src), str(dst))

    dot.render(path, format='png')

def plot_h(matrix, path, name, round="", vmin=None, vmax=None, anno=False, y_labels=None):
    submatrix = matrix[:30, :]
    round_path = path + str(round) + '.png'
    full_path = os.path.join('graph_output', round_path)
    # if not os.path.exists(full_path):
    # Plot heatmap
    if vmin != None and vmax != None:
        sns.heatmap(submatrix.cpu().detach().numpy(), annot=False, cmap='viridis', vmin=vmin, vmax=vmax)
    elif y_labels != None:
        sns.heatmap(np.array(matrix), annot=anno, cmap='viridis', yticklabels=y_labels)
    else:
        sns.heatmap(np.array(matrix), annot=False, cmap='viridis')
    plt.title(name)
    plt.savefig(full_path)
    plt.close()

def configure_plotly(x_labels, metric, name, snapshot):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_labels, y=metric, mode='lines+markers', name=name))

    fig.update_layout(
    title=f'{name} in Snapshots {snapshot}',    # Set the title of the plot
    xaxis_title='Timeline',       # Set the label for the x-axis
    yaxis_title=f'{name}',     # Set the label for the y-axis
    yaxis=dict(range=[0.4,1.0]),
    xaxis_tickangle=-45  # Rotate labels to make them more readable
   )
    
    return fig

def time_cpu(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    elapsed_time = (time.time() - start) * 1000  # Time in milliseconds

    return result, elapsed_time

def time_gpu(func, *args, **kwargs):
    """
    Measures the execution time of a GPU function using torch.cuda.Event.
    """
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()  # Ensure all operations are finished before starting
    start_event.record()

    result = func(*args, **kwargs)

    end_event.record()
    torch.cuda.synchronize()  # Ensure all operations are finished before stopping

    elapsed_time = start_event.elapsed_time(end_event)  # Time in milliseconds
    return result, elapsed_time


############## COPY METRICS TO EXCEL ###########################

# def copy_excel():
# file='stats/test.txt'

# # Extract losses
# loss_file = extract_loss_from_file(file, r'loss\s([0-9.]+)') #(file, r'loss\s*avg\s*=\s*([0-9.e-]+)')

# client_loss = [[] for _ in range(5)]

# for i, value in enumerate(loss_file):
#     client_loss[i % 5].append(value)


# """ Existing Sheet """
# df = pd.read_excel('stats/Experiments.xlsx', sheet_name='loss_gradients')
# df['Client 0 Loss'] = client_loss[0]
# df['Client 1 Loss'] = client_loss[1]
# df['Client 2 Loss'] = client_loss[2]
# df['Client 3 Loss'] = client_loss[3]
# df['Client 4 Loss'] = client_loss[4]

#     # Replace Existing Sheet
# with pd.ExcelWriter('stats/Experiments.xlsx', mode='a', if_sheet_exists='replace') as writer:
#     df.to_excel(writer, sheet_name='loss_gradients', index=False)

#     # Add on Existing Sheet
# # df.to_excel('stats/Experiments.xlsx', sheet_name='loss_gradients', index=False)

# """ New Sheet """
# # df_new = pd.DataFrame({'Client 0 Loss': client_loss[0], 'Client 1 Loss': client_loss[1], 'Client 2 Loss': client_loss[2], 'Client 3 Loss': client_loss[3], 'Client 4 Loss': client_loss[4]})
# # with pd.ExcelWriter('stats/Experiments.xlsx', mode='a') as writer:
# #     df_new.to_excel(writer, sheet_name='loss_gradients', index=False)
    
# print("Added data to excel!")


