import re
import matplotlib.pyplot as plt
import graphviz
import os
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
import time
import torch

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

def extract_edge_shapes_from_file(file_path):
    edge_index = []
    new_edge_index = []
    relevant_edge = []
    
    with open(file_path, 'r') as file:
        for line in file:
            # Extract edge_index values
            match_edge = re.search(r'ori_edge_index is of shape (\d+)', line)
            if match_edge:
                edge_index.append(int(match_edge.group(1)))
            
            # Extract new_edge_index values
            match_new_edge = re.search(r'new_edge_index is of shape (\d+)', line)
            if match_new_edge:
                new_edge_index.append(int(match_new_edge.group(1)))

            # Extract relevant_edge values
            match_relevant_edge = re.search(r'relevant_edge is of shape (\d+)', line)
            if match_relevant_edge:
                relevant_edge.append(int(match_relevant_edge.group(1)))
    
    return edge_index, new_edge_index, relevant_edge

def draw_graph(edge_index, name, round, ss, client):
    if round == 0:
        path = 'graph_output/'+'ss'+str(ss)+name+'client'+str(client)

        # if not os.path.exists(path):
        # Create a graph
        dot = graphviz.Digraph()

        # Add nodes and edges from tensor
        for i in range(edge_index.shape[1]):
            dot.edge(str(edge_index[0, i].item()), str(edge_index[1, i].item()))

        # Render the graph
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

def plot_cluster(data, label, dim, y_label=None):
    if dim == 2:
        plt.scatter(data[:, 0], data[:, 1], c=label, cmap='viridis', s=50, alpha=0.75)
        for i in range(len(data)):
            plt.text(data[i][0], data[i][1], y_label[i], ha='right')
    elif dim == 1:
        plt.scatter(data, np.zeros_like(data), c=label, cmap='viridis', s=50, alpha=0.75)
        for i in range(len(data)):
            plt.text(data[i], 0, y_label[i], ha='right')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Cluster Plot')
    plt.colorbar(label='Cluster Label')
    plt.savefig('fedassets_output/cluster.png')
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


