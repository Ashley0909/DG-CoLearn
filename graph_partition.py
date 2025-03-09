from collections import defaultdict, deque
import copy
import numpy as np
import itertools

from plot_graphs import draw_adj_list, colour_adj_list

''' Functions for Connected Components '''
def component_nodes(node, adj_list):
    queue = deque([node])
    nodes_in_component = set()

    while queue:
        current_node = queue.popleft()
        nodes_in_component.add(current_node)
        
        for neighbour in adj_list[current_node]:
            if neighbour not in nodes_in_component:
                queue.append(neighbour)

    return nodes_in_component
    
def get_all_connected_components(adj_list):
    connected_components = []
    visited = set()
    isolated_nodes = []
    
    for node in range(len(adj_list)):
        if node not in visited:
            cc = component_nodes(node, adj_list)
            if len(cc) == 1: # Isolated nodes
                isolated_nodes.append(next(iter(cc)))
            else:
                connected_components.append(cc)
            visited |= cc

    return connected_components, isolated_nodes

def connect_graphs(splittable_cc, adj_list):
    '''Link all the connected components.'''
    synthetic_edges = []
    for i, cc in enumerate(splittable_cc):
        if i == 0:
            last = list(cc)[-1] # record the last element for linking 
        else:
            # Create an undirected edge that connects the last element of previous cc and the first element of current cc
            first = list(cc)[0]
            adj_list[first].append(last)
            adj_list[last].append(first)
            synthetic_edges.append((first, last))
            synthetic_edges.append((last, first))
            last = list(cc)[-1]

    return adj_list, synthetic_edges

def count_edges(adj_list, nodes_set, synthetic_edges):
    '''Count the number of edges in a subgraph given its nodes.'''
    edge_count = 0

    for node in nodes_set:
        for neighbor in adj_list[node]:
            if neighbor in nodes_set and (node, neighbor) not in synthetic_edges:
                edge_count += 0.5  # Since edges are undirected, count half

    return edge_count

''' Functions for finding roots (by comparing shortest path of all pairs of nodes) '''
def bfs_shortest_paths(adj_list, start):
    """Computes shortest paths from `start` node to all others using BFS."""
    n = len(adj_list)
    dist = [float('inf')] * n
    dist[start] = 0
    queue = deque([start])

    while queue:
        node = queue.popleft()
        for neighbor in adj_list[node]:
            if dist[neighbor] == float('inf'):  # Unvisited
                dist[neighbor] = dist[node] + 1
                queue.append(neighbor)

    return dist

def all_pairs_shortest_paths(adj_list, isolated_nodes=[]):
    """Computes shortest paths between all pairs using BFS (for unweighted graphs)."""
    return [bfs_shortest_paths(adj_list, node) if node not in isolated_nodes else None for node in range(len(adj_list))]

def find_k_furthest_nodes(adj_list, k, isolated_nodes=[]):
    """Finds k nodes that are maximally distant from each other."""
    dist = all_pairs_shortest_paths(adj_list, isolated_nodes)
    n = len(adj_list)

    # Step 1: Find the two most distant nodes (graph diameter endpoints)
    max_dist = -1
    best_pair = None
    valid_nodes = [node for node in range(n) if node not in isolated_nodes]

    for u, v in itertools.combinations(valid_nodes, 2):
        if dist[u][v] > max_dist:
            max_dist = dist[u][v]
            best_pair = (u, v)

    selected = list(best_pair)  # Start with the two furthest nodes

    # Step 2: Iteratively select the node maximizing the minimum distance to the current set
    while len(selected) < k and len(selected) < len(valid_nodes):
        best_node = None
        max_min_dist = -1

        for node in valid_nodes:
            if node in selected:
                continue
            min_dist = min(dist[node][s] for s in selected)  # Distance to closest selected node
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_node = node

        selected.append(best_node)

    return selected[:k]

''' Function for resolving conflicting nodes '''
def resolve_conflicts(node, adj_list, global_size, subgraph_allocated, previous_level_subgraph, synthetic_edges, node_labels, isolated_nodes):
    # print("Node", node)
    # colour_adj_list(adj_list, previous_level_subgraph)
    best_subgraph, best_subgraph_score = None, 0
    for subgraph in subgraph_allocated: # compute score for given root
        number_subgraph_edges = count_edges(adj_list, list(previous_level_subgraph[subgraph] - {node} - set(isolated_nodes)), synthetic_edges)
        # print(f"Subgraph {subgraph} currently has {number_subgraph_edges} edges")
        number_neighbours = len(set(adj_list[node])&previous_level_subgraph[subgraph])
        # print("Neighbours", set(adj_list[node])&previous_level_subgraph[subgraph])
        
        if node_labels is not None: # Add labels classification score
            delta, epsilon = 0.1, 0.1
            current_subgraph = np.array(list(previous_level_subgraph[subgraph] - {node}))
            node_label_array = np.array(node_labels)
            label_counts = np.bincount(node_label_array[current_subgraph], minlength=max(node_label_array)+1)

            number_unique_labels = np.count_nonzero(label_counts)
            label_diversity_score = delta / (1+number_unique_labels)

            label_occurrence = label_counts[node_labels[node]]
            occurrence_score = epsilon / (1+label_occurrence)
            node_label_score = label_diversity_score + occurrence_score
        else:
            node_label_score = 0

        # print(f"Degree: {len(adj_list[node])} Number of Neighbours {number_neighbours}, Score {len(adj_list[node]) - number_neighbours}")
        cut_edge_score = (1/(1 + len(adj_list[node]) - number_neighbours)) # bigger => better
        edge_balanced_score = 1 - ((number_subgraph_edges)/global_size) # bigger => better
        # print(f"cut edge score {cut_edge_score}, edge score {edge_balanced_score}, node label score {node_label_score}")
        score = cut_edge_score + edge_balanced_score + node_label_score

        if score > best_subgraph_score:
            best_subgraph = subgraph
            best_subgraph_score = score

    # print(f"Node {node} allocated to Subgraph {best_subgraph}")
    return best_subgraph

''' The main function for our graph partitioning algorithm '''
def our_gpa(adj_list, global_size, node_labels=None, K=2):
    '''
    A function that splits the graph into K subgraphs which:

    1) Minimizes cut edges between subgraphs
    2) Balance Edge Counts of each subgraph
    3) Spread Node Labels evenly

    Inputs:
    1) adj_list -> List[List[]]: Adjacency List of the graph (undirected) where the ith indexed list stores the one-hop neighbours of node i 
    2) node_labels -> List[]: Node Labels of all nodes (Default as None)
    3) K -> int: Number of subgraphs

    Outputs:
    List[] -> Subgraph assignment of each node 
    '''
    previous_level_subgraph = defaultdict(set)
    node_to_allocated_subgraph = defaultdict(set)
    nodes_visited = set()

    # Identify all connected components
    connected_components, isolated_nodes = get_all_connected_components(adj_list)
    adj_list, synthetic_edges = connect_graphs(connected_components, adj_list)

    # Allocate all isolated nodes first
    for i, iso in enumerate(isolated_nodes):
        node_to_allocated_subgraph[iso] = {i%K}

    roots = find_k_furthest_nodes(adj_list, K, isolated_nodes)

    # BFS outward from each root node
    level_queue = [(root_node, i) for i, root_node in enumerate(roots)]

    while level_queue:
        nodes_visited_this_level = set()

        for node, assign in level_queue:
            node_to_allocated_subgraph[node].add(assign)
            nodes_visited_this_level.add(node)
            nodes_visited.add((node, assign))

        for node, subgraph_allocated in node_to_allocated_subgraph.items(): # First allocate the unconflicted nodes
            if len(subgraph_allocated) == 1:
                previous_level_subgraph[next(iter(subgraph_allocated))].add(node)

        # Resolve conflicts with our weighting scheme
        for node, subgraph_allocated in node_to_allocated_subgraph.items():
            if len(subgraph_allocated) > 1: # conflict case
                previous_subgraph_root = None
                for root in previous_level_subgraph:
                    if node in previous_level_subgraph[root]:
                        previous_subgraph_root = root

                best_subgraph = resolve_conflicts(node, adj_list, global_size, subgraph_allocated, previous_level_subgraph, synthetic_edges, node_labels, isolated_nodes)
                node_to_allocated_subgraph[node] &= {best_subgraph}
                if node in previous_level_subgraph[previous_subgraph_root]:
                    previous_level_subgraph[previous_subgraph_root].remove(node)
                previous_level_subgraph[best_subgraph].add(node)
        
        # Consider nodes to visit next :)
        next_level = []
        for node in nodes_visited_this_level:
            node_subgraph = next(iter(node_to_allocated_subgraph[node]))
            for neighbour in adj_list[node]:
                if neighbour not in previous_level_subgraph[node_subgraph] and (neighbour, node_subgraph) not in nodes_visited and neighbour not in roots:
                    next_level.append([neighbour, node_subgraph])

        level_queue = copy.deepcopy(next_level)

    assignment = [node_to_allocated_subgraph[i].pop() for i in range(len(node_to_allocated_subgraph))]
    # colour_adj_list(adj_list, assignment)
        
    return assignment
