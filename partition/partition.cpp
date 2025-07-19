// partition.cpp
#include "partition.hpp"
#include <optional>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <set>
#include <utility>
#include <algorithm>
#include <queue>

double count_edges(const std::unordered_set<int>& subgraph_nodes,
                   const AdjList& adj_list,
                   const std::unordered_set<std::pair<int, int>, EdgeHash>& synthetic_edges) {
    double count = 0.0;
    for (int n : subgraph_nodes) {
        for (int nb : adj_list[n]) {
            if (subgraph_nodes.count(nb) && !synthetic_edges.count({n, nb})) {
                count += 0.5;
            }
        }
    }
    return count;
}

double balance_score(int node,
    int subgraph,
    const std::unordered_map<int, std::unordered_set<int>>& previous_level_subgraph,
    const std::vector<int>& isolated_nodes,  // changed to vector
    const AdjList& adj_list,
    const std::unordered_set<std::pair<int, int>, EdgeHash>& synthetic_edges,
    int global_size) {

    std::unordered_set<int> nodes = previous_level_subgraph.at(subgraph);
    nodes.insert(node);

    std::unordered_set<int> filtered_nodes;
    for (int n : nodes) {
        // Replace isolated_nodes.count(n) with std::find for vector
        if (std::find(isolated_nodes.begin(), isolated_nodes.end(), n) == isolated_nodes.end()) {
            filtered_nodes.insert(n);
        }
    }

    int edge_count = count_edges(filtered_nodes, adj_list, synthetic_edges);

    return 1.0 - (static_cast<double>(edge_count) / global_size);
}

int refine_by_balance_and_label(
    int node,
    int current_subgraph,
    const std::unordered_set<int>& neighbor_subgraphs,
    const AdjList& adj_list,
    const std::unordered_map<int, std::unordered_set<int>>& previous_level_subgraph,
    int global_size,
    const std::unordered_set<std::pair<int, int>, EdgeHash>& synthetic_edges,
    const std::vector<int>& node_labels,
    const std::vector<int>& isolated_nodes,  // changed here
    double threshold) 
{
    int best_subgraph = current_subgraph;
    double best_improvement = 0.0;

    auto label_score = [&](int subgraph) {
        const auto& sub_nodes = previous_level_subgraph.at(subgraph);
        int label_count = 0;
        for (int n : sub_nodes) {
            if (!node_labels.empty() && node_labels[n] == node_labels[node]) {
                label_count++;
            }
        }
        return 1.0 - (label_count / (sub_nodes.size() + 1e-6));
    };

    double current_score =
        0.5 * balance_score(node, current_subgraph, previous_level_subgraph, isolated_nodes, adj_list, synthetic_edges, global_size) +
        0.5 * label_score(current_subgraph);

    for (int subgraph : neighbor_subgraphs) {
        double new_score =
            0.5 * balance_score(node, subgraph, previous_level_subgraph, isolated_nodes, adj_list, synthetic_edges, global_size) +
            0.5 * label_score(subgraph);

        if ((new_score - current_score) > std::max(threshold, best_improvement)) {
            best_subgraph = subgraph;
            best_improvement = new_score - current_score;
        }
    }

    return best_subgraph;
}


std::pair<AdjList, std::unordered_set<Edge, EdgeHash>> connect_graphs(const std::vector<std::unordered_set<int>>& splittable_cc, AdjList adj_list) {
    std::unordered_set<Edge, EdgeHash> synthetic_edges;

    if (splittable_cc.empty()) return {adj_list, synthetic_edges};

    int last = *splittable_cc[0].begin();
    for (int node : splittable_cc[0]) {
        last = node;
    }

    for (size_t i = 1; i < splittable_cc.size(); ++i) {
        const auto& cc = splittable_cc[i];

        int first = *cc.begin();
        int temp_last = first;
        for (int node : cc) temp_last = node;

        adj_list[first].push_back(last);
        adj_list[last].push_back(first);

        synthetic_edges.insert({first, last});
        synthetic_edges.insert({last, first});

        last = temp_last;
    }

    return {adj_list, synthetic_edges};
}

Component component_nodes(int node, const AdjList& adj_list) {
    std::queue<int> queue;
    Component nodes_in_component;

    queue.push(node);
    nodes_in_component.insert(node);

    while (!queue.empty()) {
        int current_node = queue.front();
        queue.pop();

        for (int neighbour : adj_list[current_node]) {
            if (nodes_in_component.find(neighbour) == nodes_in_component.end()) {
                queue.push(neighbour);
                nodes_in_component.insert(neighbour);
            }
        }
    }

    return nodes_in_component;
}

std::pair<std::vector<Component>, std::vector<int>> get_all_connected_components(const AdjList& adj_list) {
    std::vector<Component> connected_components;
    std::unordered_set<int> visited;
    std::vector<int> isolated_nodes;

    for (int node = 0; node < static_cast<int>(adj_list.size()); ++node) {
        if (visited.find(node) == visited.end()) {
            Component cc = component_nodes(node, adj_list);
            if (cc.size() == 1 && adj_list[node].empty()) {
                isolated_nodes.push_back(node);
            } else {
                connected_components.push_back(cc);
            }

            visited.insert(cc.begin(), cc.end());
        }
    }

    return {connected_components, isolated_nodes};
}

#include <vector>
#include <queue>
#include <algorithm>
#include <climits>

using AdjList = std::vector<std::vector<int>>;

std::vector<int> bfs_shortest_paths(const AdjList& adj_list, int start) {
    int n = adj_list.size();
    std::vector<int> dist(n, -1);
    std::queue<int> q;

    dist[start] = 0;
    q.push(start);

    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (int v : adj_list[u]) {
            if (dist[v] == -1) {
                dist[v] = dist[u] + 1;
                q.push(v);
            }
        }
    }

    return dist;
}

std::vector<std::vector<int>> all_pairs_shortest_paths(
    const AdjList& adj_list, const std::vector<int>& isolated_nodes) 
{
    std::vector<std::vector<int>> result;
    for (size_t node = 0; node < adj_list.size(); ++node) {
        // Check if node is isolated using std::find
        if (std::find(isolated_nodes.begin(), isolated_nodes.end(), node) != isolated_nodes.end()) {
            result.emplace_back(adj_list.size(), -1);  // all unreachable
        } else {
            result.push_back(bfs_shortest_paths(adj_list, node));
        }
    }
    return result;
}

std::vector<int> find_k_furthest_nodes(
    const AdjList& adj_list, int k, const std::vector<int>& isolated_nodes) 
{
    int n = adj_list.size();
    auto dist = all_pairs_shortest_paths(adj_list, isolated_nodes);

    // Helper lambda for checking isolated nodes
    auto is_isolated = [&](int node) {
        return std::find(isolated_nodes.begin(), isolated_nodes.end(), node) != isolated_nodes.end();
    };

    int max_dist = -1;
    std::pair<int, int> best_pair = {-1, -1};
    std::vector<int> valid_nodes;

    for (int i = 0; i < n; ++i) {
        if (!is_isolated(i)) {
            valid_nodes.push_back(i);
        }
    }

    for (int u : valid_nodes) {
        for (int v : valid_nodes) {
            if (u == v) continue;
            int d = dist[u][v];
            if (d > max_dist) {
                max_dist = d;
                best_pair = {u, v};
            }
        }
    }

    std::vector<int> selected;
    if (best_pair.first != -1) {
        selected.push_back(best_pair.first);
        if (best_pair.second != best_pair.first)
            selected.push_back(best_pair.second);
    }

    while (selected.size() < static_cast<size_t>(k) && selected.size() < valid_nodes.size()) {
        int best_node = -1;
        int max_min_dist = -1;

        for (int node : valid_nodes) {
            if (std::find(selected.begin(), selected.end(), node) != selected.end()) continue;

            int min_dist = INT_MAX;
            for (int s : selected) {
                if (dist[node][s] != -1) {
                    min_dist = std::min(min_dist, dist[node][s]);
                }
            }

            if (min_dist > max_min_dist) {
                max_min_dist = min_dist;
                best_node = node;
            }
        }

        if (best_node != -1)
            selected.push_back(best_node);
        else
            break;
    }

    if (selected.size() > static_cast<size_t>(k))
        selected.resize(k);

    return selected;
}

int resolve_by_min_cut(
    int node,
    const AdjList& adj_list,
    const std::unordered_set<int>& subgraph_allocated,
    const std::unordered_map<int, std::unordered_set<int>>& previous_level_subgraph)
{
    std::unordered_map<int, int> cut_counts;

    // Get the neighbors of the node as a set for intersection
    const auto& neighbors = adj_list[node];
    std::unordered_set<int> neighbor_set(neighbors.begin(), neighbors.end());

    for (int subgraph : subgraph_allocated) {
        const auto& sub_nodes = previous_level_subgraph.at(subgraph);
        int count = 0;

        // Count intersection size between neighbors and subgraph nodes
        for (int n : sub_nodes) {
            if (neighbor_set.count(n)) {
                count++;
            }
        }
        cut_counts[subgraph] = count;
    }

    // Find the subgraph with the maximum count
    int best_subgraph = -1;
    int max_count = -1;
    for (const auto& [subgraph, count] : cut_counts) {
        if (count > max_count) {
            max_count = count;
            best_subgraph = subgraph;
        }
    }

    return best_subgraph;
}

std::vector<int> CoLearnPartition(AdjList& adj_list, int global_size, const std::vector<int>& node_labels, int K)
{
    SubgraphToNodesMap previous_level_subgraph;
    NodeToSubgraphMap node_to_allocated_subgraph;
    std::unordered_set<int> border_nodes;

    std::set<std::pair<int, int>> nodes_visited_set;

    auto [connected_components, isolated_nodes] = get_all_connected_components(adj_list);
    auto [new_adj_list, synthetic_edges] = connect_graphs(connected_components, adj_list);
    adj_list = new_adj_list; // no move

    auto roots = find_k_furthest_nodes(adj_list, K, isolated_nodes);
    std::vector<std::pair<int, int>> level_queue;  // pair<node, assign>
    for (int i = 0; i < (int)roots.size(); ++i) {
        level_queue.emplace_back(roots[i], i);
    }

    while (!level_queue.empty()) {
        std::unordered_set<int> nodes_visited_this_level;
        for (const auto& [node, assign] : level_queue) {
            node_to_allocated_subgraph[node].insert(assign);
            nodes_visited_this_level.insert(node);
            nodes_visited_set.emplace(node, assign);
        }

        // First allocate unconflicted nodes
        for (auto& [node, subgraphs_allocated] : node_to_allocated_subgraph) {
            if (subgraphs_allocated.size() == 1) {
                int subgraph = *subgraphs_allocated.begin();
                previous_level_subgraph[subgraph].insert(node);
            }
        }

        // Resolve conflicted nodes
        for (auto& [node, subgraphs_allocated] : node_to_allocated_subgraph) {
            if (subgraphs_allocated.size() > 1) {
                int best_subgraph = resolve_by_min_cut(node, adj_list, subgraphs_allocated, previous_level_subgraph);

                node_to_allocated_subgraph[node] = {best_subgraph};
                for (int subgraph : subgraphs_allocated) {
                    previous_level_subgraph[subgraph].erase(node);
                }
                previous_level_subgraph[best_subgraph].insert(node);
                border_nodes.insert(node);
            }
        }

        // Prepare next level
        std::vector<std::pair<int, int>> next_level;
        for (int node : nodes_visited_this_level) {
            int node_subgraph = *node_to_allocated_subgraph[node].begin();
            for (int neighbour : adj_list[node]) {
                if (nodes_visited_set.find({neighbour, node_subgraph}) == nodes_visited_set.end() &&
                    std::find(roots.begin(), roots.end(), neighbour) == roots.end()) {

                    next_level.emplace_back(neighbour, node_subgraph);
                    if (node_to_allocated_subgraph.count(neighbour) &&
                        node_to_allocated_subgraph[neighbour].size() == 1) {

                        if (*node_to_allocated_subgraph[neighbour].begin() != node_subgraph) {
                            border_nodes.insert(neighbour);
                        }
                    }
                }
            }
        }
        level_queue = std::move(next_level);
    }

    // Refinement for border nodes
    for (int node : border_nodes) {
        int current_subgraph = *node_to_allocated_subgraph[node].begin();
        const auto& neighbors = adj_list[node];
        SubgraphSet neighbor_subgraphs;
        for (int n : neighbors) {
            if (node_to_allocated_subgraph.count(n)) {
                neighbor_subgraphs.insert(*node_to_allocated_subgraph[n].begin());
            }
        }

        int best_subgraph = refine_by_balance_and_label(node, current_subgraph, neighbor_subgraphs,
            adj_list, previous_level_subgraph, global_size, synthetic_edges, node_labels, isolated_nodes);

        if (best_subgraph != current_subgraph) {
            previous_level_subgraph[current_subgraph].erase(node);
            previous_level_subgraph[best_subgraph].insert(node);
            node_to_allocated_subgraph[node] = {best_subgraph};
        }
    }

    // Assign isolated nodes
    for (int node : isolated_nodes) {
        int best_subgraph = 0;
        int best_diversity = -1;

        for (const auto& [subgraph, nodes] : previous_level_subgraph) {
            if (!node_labels.empty()) {
                std::vector<int> labels;
                for (int n : nodes) {
                    labels.push_back(node_labels[n]); // FIXED
                }
                std::unordered_set<int> unique_labels(labels.begin(), labels.end());
                int diversity = (int)unique_labels.size();
                if (diversity > best_diversity) {
                    best_diversity = diversity;
                    best_subgraph = subgraph;
                }
            }
        }
        node_to_allocated_subgraph[node] = {best_subgraph};
        previous_level_subgraph[best_subgraph].insert(node);
    }

    // Convert assignment map to vector
    int max_node = 0;
    for (const auto& [node, _] : node_to_allocated_subgraph) {
        if (node > max_node) max_node = node;
    }
    std::vector<int> assignment(max_node + 1, -1);
    for (const auto& [node, subgraph_set] : node_to_allocated_subgraph) {
        assignment[node] = *subgraph_set.begin();
    }

    return assignment;
}
