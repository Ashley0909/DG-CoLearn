""" Helper Function """
def count_edges(subgraph_nodes, adj_list, synthetic_edges):
        count = 0
        for n in subgraph_nodes:
            for nb in adj_list[n]:
                if nb in subgraph_nodes and (n, nb) not in synthetic_edges:
                    count += 0.5
        return count

def balance_score(node, subgraph, previous_level_subgraph, isolated_nodes, adj_list, synthetic_edges, global_size):
        nodes = previous_level_subgraph[subgraph] | {node}
        edge_count = count_edges(nodes - set(isolated_nodes), adj_list, synthetic_edges)
        return 1 - (edge_count / global_size)

""" Primary Resolve """
def resolve_by_min_cut(node, adj_list, subgraph_allocated, previous_level_subgraph):
    cut_counts = {
        subgraph: len(set(adj_list[node]) & previous_level_subgraph[subgraph])
        for subgraph in subgraph_allocated
    }
    return max(cut_counts, key=cut_counts.get)

def resolve_by_edge_balance(node, adj_list, subgraph_allocated, previous_level_subgraph, global_size, synthetic_edges, isolated_nodes):
    edge_scores = {}
    for subgraph in subgraph_allocated:
        temp_nodes = previous_level_subgraph[subgraph] - {node} - set(isolated_nodes)
        edge_count = count_edges(temp_nodes, adj_list, synthetic_edges)
        score = 1 - (edge_count / global_size)
        edge_scores[subgraph] = score

    return max(edge_scores, key=edge_scores.get)

def resolve_by_label_balance(node, subgraph_allocated, previous_level_subgraph, node_labels):
    if node_labels == None:
        return 0
    label_scores = {}
    for subgraph in subgraph_allocated:
        sub_nodes = list(previous_level_subgraph[subgraph] - {node})
        label_count = sum(1 for n in sub_nodes if node_labels[n] == node_labels[node])
        label_scores[subgraph] = 1 - (label_count / (len(sub_nodes) + 1e-6))
    return max(label_scores, key=label_scores.get)

""" Refinement Functions """
def refine_by_balance_and_label(node, current_subgraph, neighbor_subgraphs, adj_list, previous_level_subgraph, global_size, synthetic_edges, node_labels, isolated_nodes, threshold=0.5):
    best_subgraph = current_subgraph
    best_improvement = 0

    def label_score(subgraph):
        sub_nodes = list(previous_level_subgraph[subgraph])
        label_count = sum(1 for n in sub_nodes if node_labels[n] == node_labels[node]) if node_labels != None else 0
        return 1 - (label_count / (len(sub_nodes) + 1e-6))

    current_score = 0.5 * balance_score(node, current_subgraph, previous_level_subgraph, isolated_nodes, adj_list, synthetic_edges, global_size) + 0.5 * label_score(current_subgraph)

    for subgraph in neighbor_subgraphs:
        new_score = 0.5 * balance_score(node, subgraph, previous_level_subgraph, isolated_nodes, adj_list, synthetic_edges, global_size) + 0.5 * label_score(subgraph)
        if new_score - current_score > max(threshold, best_improvement):
            best_subgraph = subgraph  
            best_improvement = new_score - current_score

    return best_subgraph

def refine_by_min_cut_and_label(node, current_subgraph, neighbor_subgraphs, adj_list, previous_level_subgraph, node_labels, threshold=0.5):
    current_cut = len(set(adj_list[node]) & previous_level_subgraph[current_subgraph])
    current_label_score = sum(1 for n in previous_level_subgraph[current_subgraph] if node_labels[n] == node_labels[node]) if node_labels != None else 0
    current_score = 0.5 * current_cut + 0.5 * current_label_score

    best_subgraph = current_subgraph
    best_improvement = 0

    for subgraph in neighbor_subgraphs:
        cut = len(set(adj_list[node]) & previous_level_subgraph[subgraph])
        label_score = sum(1 for n in previous_level_subgraph[subgraph] if node_labels[n] == node_labels[node]) if node_labels != None else 0
        new_score = 0.5 * cut + 0.5 * label_score
        if new_score - current_score > max(threshold, best_improvement):
            best_subgraph = subgraph
            best_improvement = new_score - current_score

    return best_subgraph

def refine_by_min_cut_and_balance(node, current_subgraph, neighbor_subgraphs, adj_list, previous_level_subgraph, global_size, synthetic_edges, isolated_nodes, threshold=0.5):
    def cut_score(subgraph):
        return len(set(adj_list[node]) & previous_level_subgraph[subgraph])

    current_score = 0.5 * cut_score(current_subgraph) + 0.5 * balance_score(node, current_subgraph, previous_level_subgraph, isolated_nodes, adj_list, synthetic_edges, global_size)
    best_subgraph = current_subgraph
    best_improvement = 0

    for subgraph in neighbor_subgraphs:
        new_score = 0.5 * cut_score(subgraph) + 0.5 * balance_score(node, subgraph, previous_level_subgraph, isolated_nodes, adj_list, synthetic_edges, global_size)
        if new_score - current_score > max(threshold, best_improvement):
            best_subgraph = subgraph
            best_improvement = new_score - current_score

    return best_subgraph
