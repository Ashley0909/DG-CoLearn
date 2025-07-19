// partition.hpp
#ifndef PARTITION_HPP
#define PARTITION_HPP

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <set>
#include <utility>

// Type aliases for convenience
using AdjList = std::vector<std::vector<int>>;
using SubgraphSet = std::unordered_set<int>;
using NodeToSubgraphMap = std::unordered_map<int, SubgraphSet>;
using SubgraphToNodesMap = std::unordered_map<int, SubgraphSet>;
using Component = std::unordered_set<int>;
using Edge = std::pair<int, int>;

// Hash struct for Edge (pair<int, int>)
struct EdgeHash {
    std::size_t operator()(const Edge& e) const noexcept {
        return std::hash<int>()(e.first) ^ (std::hash<int>()(e.second) << 1);
    }
};

// Function declarations

// Count edges inside a subgraph, excluding synthetic edges
double count_edges(const SubgraphSet& subgraph_nodes,
                   const AdjList& adj_list,
                   const std::unordered_set<Edge, EdgeHash>& synthetic_edges);

// Calculate balance score for a node w.r.t. a subgraph
double balance_score(int node,
                     int subgraph,
                     const std::unordered_map<int, std::unordered_set<int>>& previous_level_subgraph,
                     const std::vector<int>& isolated_nodes,
                     const AdjList& adj_list,
                     const std::unordered_set<Edge, EdgeHash>& synthetic_edges,
                     int global_size);

// Refine node assignment by balancing label similarity and subgraph balance
int refine_by_balance_and_label(int node,
                                int current_subgraph,
                                const std::unordered_set<int>& neighbor_subgraphs,
                                const AdjList& adj_list,
                                const std::unordered_map<int, std::unordered_set<int>>& previous_level_subgraph,
                                int global_size,
                                const std::unordered_set<Edge, EdgeHash>& synthetic_edges,
                                const std::vector<int>& node_labels,
                                const std::vector<int>& isolated_nodes,
                                double threshold = 0.0);

// Connect disconnected components of a graph by adding synthetic edges
std::pair<AdjList, std::unordered_set<Edge, EdgeHash>>
connect_graphs(const std::vector<std::unordered_set<int>>& splittable_cc, AdjList adj_list);

// Find all nodes connected to a given node (connected component)
Component component_nodes(int node, const AdjList& adj_list);

// Get all connected components and isolated nodes in a graph
std::pair<std::vector<Component>, std::vector<int>>
get_all_connected_components(const AdjList& adj_list);

// Compute shortest distances from start node to all others using BFS
std::vector<int> bfs_shortest_paths(const AdjList& adj_list, int start);

// Compute all pairs shortest paths, respecting isolated nodes
std::vector<std::vector<int>> all_pairs_shortest_paths(
    const AdjList& adj_list,
    const std::vector<int>& isolated_nodes = {}
);

// Find k nodes that are furthest apart (heuristic)
std::vector<int> find_k_furthest_nodes(
    const AdjList& adj_list,
    int k,
    const std::vector<int>& isolated_nodes = {}
);

// Resolve assignment for a node by minimum cut heuristic
int resolve_by_min_cut(int node,
                       const AdjList& adj_list,
                       const std::unordered_set<int>& subgraph_allocated,
                       const std::unordered_map<int, std::unordered_set<int>>& previous_level_subgraph);

// Main partitioning function
std::vector<int> CoLearnPartition(AdjList& adj_list,
                                  int global_size,
                                  const std::vector<int>& node_labels,
                                  int K);

#endif // PARTITION_HPP
