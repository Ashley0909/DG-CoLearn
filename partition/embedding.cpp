#include <optional>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <set>
#include <utility>
#include <algorithm>
#include <queue>
#include <functional>

using EmbeddingTensor = std::vector<std::vector<std::vector<std::vector<float>>>>;

std::vector<std::vector<std::vector<float>>> fast_get_global_embedding_cpp(
    const EmbeddingTensor& embeddings,
    const std::vector<std::vector<int>>& adj_list,
    const std::vector<std::vector<std::tuple<int, int, int>>>& precomputed_ne_needed,
    const std::vector<int>& node_assignment,
    const std::vector<int>& subnodes_union
) {
    int num_hops = 3;
    int num_nodes = adj_list.size();
    std::vector<std::vector<std::vector<float>>> hop_embeddings;

    std::unordered_map<int, bool> subnode_set;
    for (int node : subnodes_union)
        subnode_set[node] = true;

    for (int hop = 0; hop < num_hops; ++hop) {
        std::vector<std::vector<float>> hop_matrix;
        for (int node = 0; node < num_nodes; ++node) {
            int client_id = node_assignment[node];
            std::vector<float> final_embedding = embeddings[client_id][hop][node];

            // Check if this node needs neighbor embedding aggregation
            if (!precomputed_ne_needed[node].empty() && hop > 0) {
                for (const auto& [client, neighbor, count] : precomputed_ne_needed[node]) {
                    if (subnode_set.find(neighbor) != subnode_set.end()) {
                        const std::vector<float>& neighbor_emb = embeddings[client][hop - 1][neighbor];
                        for (size_t i = 0; i < final_embedding.size(); ++i) {
                            final_embedding[i] += neighbor_emb[i] * count;
                        }
                    }
                }
            }

            hop_matrix.push_back(final_embedding);
        }

        hop_embeddings.push_back(hop_matrix);
    }

    return hop_embeddings;
}