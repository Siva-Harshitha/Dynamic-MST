#include <iostream>
#include <vector>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include <queue>
#include <iomanip>
#include <cassert>
#include <functional>

using namespace std;

// Hash function for pairs
struct pair_hash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

// Euler Tour Node
struct ETNode {
    char vertex;
    ETNode* left;
    ETNode* right;
    ETNode* parent;
    int size;
    double max_weight; // Max weight on subtree path
    pair<char, char> max_edge; // Edge with max weight
    vector<pair<char, char>> nontree_edges; // Non-tree edges incident to vertex

    ETNode(char v) : vertex(v), left(nullptr), right(nullptr), parent(nullptr), size(1),
                     max_weight(0.0), max_edge({'\0', '\0'}) {}
};

// Euler Tour Tree
class ETTree {
private:
    ETNode* root;

    void update(ETNode* node) {
        if (!node) return;
        node->size = 1 + (node->left ? node->left->size : 0) + (node->right ? node->right->size : 0);
        node->max_weight = 0.0;
        node->max_edge = {'\0', '\0'};
        if (node->left && node->left->max_weight > node->max_weight) {
            node->max_weight = node->left->max_weight;
            node->max_edge = node->left->max_edge;
        }
        if (node->right && node->right->max_weight > node->max_weight) {
            node->max_weight = node->right->max_weight;
            node->max_edge = node->right->max_edge;
        }
    }

    void rotate_right(ETNode* x) {
        ETNode* y = x->left;
        x->left = y->right;
        if (y->right) y->right->parent = x;
        y->parent = x->parent;
        if (!x->parent) root = y;
        else if (x == x->parent->right) x->parent->right = y;
        else x->parent->left = y;
        y->right = x;
        x->parent = y;
        update(x);
        update(y);
    }

    void rotate_left(ETNode* x) {
        ETNode* y = x->right;
        x->right = y->left;
        if (y->left) y->left->parent = x;
        y->parent = x->parent;
        if (!x->parent) root = y;
        else if (x == x->parent->left) x->parent->left = y;
        else x->parent->right = y;
        y->left = x;
        x->parent = y;
        update(x);
        update(y);
    }

    void splay(ETNode* x) {
        while (x->parent) {
            ETNode* p = x->parent;
            ETNode* g = p->parent;
            if (g) {
                if (p == g->left) {
                    if (x == p->left) {
                        rotate_right(g);
                        rotate_right(p);
                    } else {
                        rotate_left(p);
                        rotate_right(g);
                    }
                } else {
                    if (x == p->right) {
                        rotate_left(g);
                        rotate_left(p);
                    } else {
                        rotate_right(p);
                        rotate_left(g);
                    }
                }
            } else {
                if (x == p->left) rotate_right(p);
                else rotate_left(p);
            }
        }
    }

    ETNode* find_node(char vertex, ETNode* node) {
        if (!node) return nullptr;
        if (node->vertex == vertex) {
            splay(node);
            return node;
        }
        if (vertex < node->vertex && node->left) return find_node(vertex, node->left);
        if (vertex > node->vertex && node->right) return find_node(vertex, node->right);
        return nullptr;
    }

public:
    ETTree() : root(nullptr) {}

    void insert(char vertex) {
        ETNode* new_node = new ETNode(vertex);
        if (!root) {
            root = new_node;
            return;
        }
        ETNode* curr = root;
        ETNode* parent = nullptr;
        while (curr) {
            parent = curr;
            if (vertex < curr->vertex) curr = curr->left;
            else if (vertex > curr->vertex) curr = curr->right;
            else {
                delete new_node;
                return;
            }
        }
        if (vertex < parent->vertex) parent->left = new_node;
        else parent->right = new_node;
        new_node->parent = parent;
        splay(new_node);
        update(parent);
        update(new_node);
    }

    ETNode* find(char vertex) {
        return find_node(vertex, root);
    }

    void add_nontree_edge(char vertex, pair<char, char> edge) {
        ETNode* node = find(vertex);
        if (node) {
            node->nontree_edges.push_back(edge);
            update(node);
        }
    }

    void remove_nontree_edge(char vertex, pair<char, char> edge) {
        ETNode* node = find(vertex);
        if (node) {
            auto it = find_if(node->nontree_edges.begin(), node->nontree_edges.end(),
                              [&edge](const auto& e) { return e == edge; });
            if (it != node->nontree_edges.end()) {
                node->nontree_edges.erase(it);
                update(node);
            }
        }
    }

    void update_edge(char u, char v, double weight) {
        ETNode* node = find(u);
        if (node) {
            pair<char, char> edge = {min(u, v), max(u, v)};
            if (weight > node->max_weight) {
                node->max_weight = weight;
                node->max_edge = edge;
            }
            splay(node);
        }
    }

    pair<double, pair<char, char>> get_max_edge(char u, char v) {
        ETNode* node_u = find(u);
        if (!node_u) return {0.0, {'\0', '\0'}};
        ETNode* node_v = find(v);
        if (!node_v) return {0.0, {'\0', '\0'}};
        // Splay v to root, check max_edge
        double max_weight = node_v->max_weight;
        pair<char, char> max_edge = node_v->max_edge;
        return {max_weight, max_edge};
    }

    pair<ETTree, ETTree> split(char vertex) {
        ETNode* node = find(vertex);
        if (!node) return {ETTree(), ETTree()};
        splay(node);
        ETTree left_tree, right_tree;
        left_tree.root = node->left;
        right_tree.root = node->right;
        if (node->left) node->left->parent = nullptr;
        if (node->right) node->right->parent = nullptr;
        node->left = node->right = nullptr;
        update(node);
        left_tree.insert(node->vertex);
        delete node;
        return {left_tree, right_tree};
    }

    void join(ETTree& other) {
        if (!other.root) return;
        if (!root) {
            root = other.root;
            other.root = nullptr;
            return;
        }
        ETNode* max_node = root;
        while (max_node->right) max_node = max_node->right;
        splay(max_node);
        max_node->right = other.root;
        if (other.root) other.root->parent = max_node;
        update(max_node);
        other.root = nullptr;
    }

    vector<pair<char, char>> get_nontree_edges() {
        vector<pair<char, char>> result;
        function<void(ETNode*)> collect = [&](ETNode* node) {
            if (!node) return;
            result.insert(result.end(), node->nontree_edges.begin(), node->nontree_edges.end());
            collect(node->left);
            collect(node->right);
        };
        collect(root);
        return result;
    }

    vector<char> get_vertices() {
        vector<char> result;
        function<void(ETNode*)> collect = [&](ETNode* node) {
            if (!node) return;
            collect(node->left);
            result.push_back(node->vertex);
            collect(node->right);
        };
        collect(root);
        return result;
    }
};

// Disjoint Set
class DisjointSet {
private:
    unordered_map<char, char> parent;
    unordered_map<char, int> rank;
public:
    void make_set(char x) {
        if (parent.find(x) == parent.end()) {
            parent[x] = x;
            rank[x] = 0;
        }
    }
    char find(char x) {
        if (parent.find(x) == parent.end()) make_set(x);
        if (parent[x] != x) parent[x] = find(parent[x]);
        return parent[x];
    }
    bool union_sets(char x, char y) {
        char root_x = find(x);
        char root_y = find(y);
        if (root_x == root_y) return false;
        if (rank[root_x] < rank[root_y]) parent[root_x] = root_y;
        else if (rank[root_x] > rank[root_y]) parent[root_y] = root_x;
        else {
            parent[root_y] = root_x;
            rank[root_x]++;
        }
        return true;
    }
    bool same_component(char u, char v) {
        return find(u) == find(v);
    }
    void reset() {
        parent.clear();
        rank.clear();
    }
    void debug_print() {
        cout << "DisjointSet state: ";
        for (const auto& [node, par] : parent) {
            cout << node << "->parent:" << par << ",rank:" << rank[node] << " ";
        }
        cout << endl;
    }
};

// Dynamic MST
class DynamicMST {
private:
    vector<ETTree> level_forests; // Spanning forests for G_i
    vector<set<pair<char, char>>> level_edges; // E_i
    unordered_map<pair<char, char>, double, pair_hash> edge_weights;
    unordered_map<pair<char, char>, bool, pair_hash> is_tree_edge;
    unordered_map<char, unordered_set<char>> adj_list;
    set<pair<char, char>> edges;
    DisjointSet ds;
    set<int> active_levels; // Balanced binary tree for levels
    int max_levels;
    double epsilon;
    double min_weight;
    double max_weight;
    int n;

    void initialize_levels(int nodes, double eps) {
        n = nodes;
        epsilon = eps;
        min_weight = 1.0;
        max_weight = 1.0;
        max_levels = ceil(log(max_weight / min_weight + 1e-10) / log(1 + epsilon)) + 1;
        level_forests.resize(max_levels);
        level_edges.resize(max_levels);
        active_levels.insert(0); //Adds level 0 to active_levels
    }

    int get_level(double weight) {  //Determines the level of an edge based on its weight
        if (weight < min_weight) return -1;
        return floor(log(weight / min_weight + 1e-10) / log(1 + epsilon));
    }

    void resize_levels(int new_max_levels) {
        if (new_max_levels <= max_levels) return;
        vector<pair<pair<char, char>, double>> all_edges;
        for (const auto& e : edges) {   //Collects all edges and their weights.
            all_edges.emplace_back(e, edge_weights[e]);
        }
        ds.reset();
        level_forests.clear();
        level_edges.clear();
        active_levels.clear();
        level_forests.resize(new_max_levels);
        level_edges.resize(new_max_levels);
        max_levels = new_max_levels;
        unordered_map<pair<char, char>, bool, pair_hash> old_tree_edges = is_tree_edge;
        is_tree_edge.clear();
        for (const auto& [e, w] : all_edges) {
            int lvl = get_level(w);            //Computes their new levels
            if (lvl >= 0 && lvl < max_levels) {  //Reinserts the edge
                edges.insert(e);
                edge_weights[e] = w;
                level_edges[lvl].insert(e);
                active_levels.insert(lvl);
                adj_list[e.first].insert(e.second);
                adj_list[e.second].insert(e.first);
                is_tree_edge[e] = old_tree_edges[e];
                if (is_tree_edge[e]) {
                    ds.union_sets(e.first, e.second);
                    for (int j = lvl; j < max_levels; ++j) {
                        level_forests[j].update_edge(e.first, e.second, w);
                    }
                } else {
                    for (int j = lvl; j < max_levels; ++j) {
                        level_forests[j].add_nontree_edge(e.first, e);
                        level_forests[j].add_nontree_edge(e.second, e);
                    }
                }
            }
        }
    }

    pair<char, char> find_max_cycle_edge(char u, char v, double new_weight) {
        if (!ds.same_component(u, v)) return {'\0', '\0'};  //null edge
        // Use level_forests[0] as F
        auto [weight, max_edge] = level_forests[0].get_max_edge(u, v);
        if (weight <= new_weight || max_edge.first == '\0') return {'\0', '\0'};
        return max_edge;
    }

    pair<char, char> find_replacement_edge(char u, char v, int& min_level) {
        pair<char, char> replacement = {'\0', '\0'};
        double min_weight = numeric_limits<double>::max();
        min_level = -1;
        // Binary search over active levels find the lowest level i where u and v are in 
        //different components and a suitable non-tree edge exists
        vector<int> levels(active_levels.begin(), active_levels.end());
        int left = 0, right = levels.size() - 1;
        while (left <= right) {
            int mid = (left + right) / 2;
            int i = levels[mid];
            vector<char> vertices = level_forests[i].get_vertices();
            if (find(vertices.begin(), vertices.end(), u) != vertices.end() &&
                find(vertices.begin(), vertices.end(), v) != vertices.end() &&
                !ds.same_component(u, v)) {
                // Check non-tree edges in level_edges[i]
                for (const auto& e : level_edges[i]) {
                    if (!is_tree_edge[e] && edge_weights[e] < min_weight) {
                        DisjointSet temp_ds = ds;
                        temp_ds.union_sets(e.first, e.second);
                        if (temp_ds.same_component(u, v)) {   //Updates min_level to the
                                                            //level of the selected edge.
                            min_weight = edge_weights[e];
                            replacement = e;
                            min_level = i;
                        }
                    }
                }
                right = mid - 1; // Try lower levels
            } else {
                left = mid + 1; // Need higher levels
            }
        }
        return replacement;
    }

public:
    DynamicMST(int nodes, double eps) : n(nodes), epsilon(eps), min_weight(1.0), max_weight(1.0) {
        initialize_levels(nodes, eps);
    }

    void add_node(char node) {
        if (adj_list.find(node) == adj_list.end()) {
            adj_list[node] = unordered_set<char>();     //creates an empty adjacency set
            ds.make_set(node);
            for (int i : active_levels) {
                level_forests[i].insert(node);
            }
        }
    }

    bool add_edge(char u, char v, double weight) {
        if (u == v) return false;
        if (u > v) swap(u, v);
        pair<char, char> edge = {u, v};
        if (edges.count(edge)) return false;

        max_weight = max(max_weight, weight);
        int new_max_levels = ceil(log(max_weight / min_weight + 1e-10) / log(1 + epsilon)) + 1;
        resize_levels(new_max_levels);

        int level = get_level(weight);
        if (level < 0 || level >= max_levels) return false;

        edges.insert(edge);
        edge_weights[edge] = weight;
        adj_list[u].insert(v);
        adj_list[v].insert(u);
        level_edges[level].insert(edge);
        active_levels.insert(level);

        add_node(u);
        add_node(v);

        bool is_new_component = !ds.same_component(u, v);
        cout << "Adding edge (" << u << "," << v << "," << weight << "): components same? " << !is_new_component << endl;

        if (is_new_component) {
            is_tree_edge[edge] = true;
            ds.union_sets(u, v);
            for (int j : active_levels) {
                level_forests[j].update_edge(u, v, weight);
            }
        } else {
            pair<char, char> max_edge = find_max_cycle_edge(u, v, weight);
            if (max_edge.first != '\0' && weight < edge_weights[max_edge]) {
                cout << "Replacing edge (" << max_edge.first << "," << max_edge.second << "," << edge_weights[max_edge] << ") with (" << u << "," << v << "," << weight << ")" << endl;
                is_tree_edge[max_edge] = false;
                int max_level = get_level(edge_weights[max_edge]);
                level_edges[max_level].insert(max_edge);
                for (int j : active_levels) {
                    if (j >= max_level) {
                        level_forests[j].add_nontree_edge(max_edge.first, max_edge);
                        level_forests[j].add_nontree_edge(max_edge.second, max_edge);
                    }
                }
                is_tree_edge[edge] = true;
                for (int j : active_levels) {
                    level_forests[j].update_edge(u, v, weight);
                }
            } else {
                is_tree_edge[edge] = false;
                for (int j : active_levels) {
                    if (j >= level) {
                        level_forests[j].add_nontree_edge(u, edge);
                        level_forests[j].add_nontree_edge(v, edge);
                    }
                }
            }
        }

        double mst_weight = get_mst_weight(true);
        double approx_weight = get_mst_weight(false);
        if (approx_weight > mst_weight * (1 + epsilon)) {
            cerr << "Approximation guarantee violated: approx_weight=" << approx_weight << " > " << mst_weight * (1 + epsilon) << endl;
            assert(false);
        }

        return true;
    }

    bool remove_edge(char u, char v, int& level) {
        if (u > v) swap(u, v);
        pair<char, char> edge = {u, v};
        if (!edges.count(edge)) return false;

        level = get_level(edge_weights[edge]);
        if (level < 0 || level >= max_levels) return false;

        edges.erase(edge);
        adj_list[u].erase(v);
        adj_list[v].erase(u);
        level_edges[level].erase(edge);
        if (level_edges[level].empty()) active_levels.erase(level);

        bool is_tree = is_tree_edge[edge];
        double w_e = edge_weights[edge];
        edge_weights.erase(edge);
        is_tree_edge.erase(edge);

        if (is_tree) {
            cout << "Removing tree edge (" << u << "," << v << "," << w_e << ") from level " << level << endl;
            ds.reset();     //rebuilds components using remaining tree edges
            for (const auto& e : edges) {
                if (is_tree_edge[e]) {
                    ds.union_sets(e.first, e.second);
                }
            }
            //cout << "After resetting connectivity: ";
            //ds.debug_print();

            int min_level;
            pair<char, char> replacement = find_replacement_edge(u, v, min_level);
            if (replacement.first != '\0') {
                cout << "Selected replacement edge (" << replacement.first << "," << replacement.second << "," << edge_weights[replacement] << ")" << endl;
                is_tree_edge[replacement] = true;
                ds.union_sets(replacement.first, replacement.second);
                for (int j : active_levels) {
                    if (j >= min_level) {
                        level_forests[j].update_edge(replacement.first, replacement.second, edge_weights[replacement]);
                        level_forests[j].remove_nontree_edge(replacement.first, replacement);
                        level_forests[j].remove_nontree_edge(replacement.second, replacement);
                    }
                }
                active_levels.insert(min_level);
            } else {
                cout << "No replacement edge found" << endl;
            }
        } else {
            cout << "Removing non-tree edge (" << u << "," << v << "," << w_e << ") from level " << level << endl;
            for (int j : active_levels) {
                if (j >= level) {
                    level_forests[j].remove_nontree_edge(u, edge);
                    level_forests[j].remove_nontree_edge(v, edge);
                }
            }
        }

        double mst_weight = get_mst_weight(true);
        double approx_weight = get_mst_weight(false);
        if (approx_weight > mst_weight * (1 + epsilon)) {
            cerr << "Approximation guarantee violated: approx_weight=" << approx_weight << " > " << mst_weight * (1 + epsilon) << endl;
            assert(false);
        }

        return true;
    }

    bool same_component(char u, char v) {
        return ds.same_component(u, v);
    }

    double get_mst_weight(bool print = true) {
        DisjointSet temp_ds;
        vector<pair<pair<char, char>, double>> sorted_edges;
        for (const auto& e : edges) {
            sorted_edges.emplace_back(e, edge_weights[e]);
        }
        //sorts edges by weight.
        sort(sorted_edges.begin(), sorted_edges.end(), [](const auto& a, const auto& b) {
            return a.second < b.second;
        });
        //find the exact MST weight and edges.
        double mst_weight = 0;
        vector<pair<char, char>> mst_edges;
        for (const auto& [e, w] : sorted_edges) {
            if (temp_ds.union_sets(e.first, e.second)) {
                mst_edges.push_back(e);
                mst_weight += w;
            }
        }
//For the approximate weight, assigns each MST edge the weight
//of its level (min_weight * (1+ε)^level).
        double approx_mst_weight = 0;
        vector<pair<char, char>> approx_mst_edges;
        for (const auto& e : mst_edges) {
            approx_mst_edges.push_back(e);
            int level = get_level(edge_weights[e]);
            double level_weight = min_weight * pow(1 + epsilon, level);
            approx_mst_weight += level_weight;
        }
        sort(approx_mst_edges.begin(), approx_mst_edges.end());

        if (print) {
            cout << "Exact MST edges: ";
            for (const auto& e : mst_edges) {
                cout << "(" << e.first << "," << e.second << "," << fixed << setprecision(4) << edge_weights[e] << ") ";
            }
            cout << "[Total: " << fixed << setprecision(4) << mst_weight << "]" << endl;

            cout << "(1+epsilon)-MST edges (epsilon=" << epsilon << "): ";
            for (const auto& e : approx_mst_edges) {
                int level = get_level(edge_weights[e]);
                double level_weight = min_weight * pow(1 + epsilon, level);
                cout << "(" << e.first << "," << e.second << "," << fixed << setprecision(4) << level_weight << ") ";
            }
            cout << "[Total: " << fixed << setprecision(4) << approx_mst_weight << "]" << endl;
            cout << "Approximation guarantee: weight <= " << fixed << setprecision(4) << mst_weight * (1 + epsilon) << endl;
        }

        return print ? mst_weight : approx_mst_weight;
    }

    void display_level_edges() {
        cout << "Edges by MST level (original weights):" << endl;
    
        for (int i : active_levels) { //For each active level i
            // Computes the weight range [min_weight * (1+ε)^i, min_weight * (1+ε)^(i+1)).
            double lower_bound = min_weight * pow(1 + epsilon, i);
            double upper_bound = min_weight * pow(1 + epsilon, i + 1);
            cout << "Level " << i << " (weights [" << fixed << setprecision(4) << lower_bound << ", " << upper_bound << ")): {";
            vector<pair<char, char>> tree_edges;
            for (const auto& e : level_edges[i]) { // Prints tree edges in level_edges[i] with original weights.
                if (is_tree_edge[e]) {
                    tree_edges.push_back(e);
                }
            }
            sort(tree_edges.begin(), tree_edges.end());
            for (const auto& e : tree_edges) {
                cout << "(" << e.first << ", " << e.second << ", " << fixed << setprecision(4) << edge_weights[e] << ") ";
            }
            cout << "}" << endl;
        }
        //Repeats for approximated weights,
        cout << "\nEdges by MST level ((1+epsilon)-approximated weights):" << endl;
        for (int i : active_levels){//using min_weight * (1+ε)^i for all edges in level i.
            double level_weight = min_weight * pow(1 + epsilon, i);
            cout << "Level " << i << " (weight = " << fixed << setprecision(4) << level_weight << "): {";
            vector<pair<char, char>> tree_edges;
            for (const auto& e : level_edges[i]) {// Prints tree edges in level_edges[i] 
                                                    //with approximated weights.
                if (is_tree_edge[e]) {
                    tree_edges.push_back(e);
                }
            }
            sort(tree_edges.begin(), tree_edges.end());
            for (const auto& e : tree_edges) {
                cout << "(" << e.first << ", " << e.second << ", " << fixed << setprecision(4) << level_weight << ") ";
            }
            cout << "}" << endl;
        }
    }

    vector<char> euler_tour(char vertex) {
        vector<char> tour;
        if (adj_list.find(vertex) == adj_list.end()) return tour;

        unordered_set<char> visited;
        cout << "Building Euler tour from " << vertex << " using tree edges:" << endl;
        function<void(char, char)> dfs = [&](char v, char parent) {
            tour.push_back(v);
            visited.insert(v);
            cout << "Visiting " << v << endl;
            for (char u : adj_list[v]) {0
                pair<char, char> edge = {min(v, u), max(v, u)};
                if (is_tree_edge[edge] && u != parent) {
                    cout << "  Traversing tree edge (" << v << "," << u << "," << edge_weights[edge] << ")" << endl;
                    dfs(u, v);
                    tour.push_back(v);
                    cout << "Returning to " << v << endl;
                }
            }
        };
        dfs(vertex, '\0');
        return tour;
    }
};

void run_interactive_demo() {
    cout << "Enter number of nodes and epsilon: ";
    int n;
    double epsilon;
    cin >> n >> epsilon;
    if (n > 26) {
        cout << "Number of nodes cannot exceed 26 (a-z)" << endl;
        return;
    }
    DynamicMST mst(n, epsilon);
    cout << "Interactive (1+epsilon)-MST Demo" << endl;
    cout << "Commands:" << endl;
    cout << "  add_node <node>: Add a node (lowercase letter a-z)" << endl;
    cout << "  add_edge <u> <v> <weight>: Add edge (u,v) with weight" << endl;
    cout << "  remove_edge <u> <v>: Remove edge (u,v)" << endl;
    cout << "  connected <u> <v>: Check if nodes u and v are connected" << endl;
    cout << "  mst_weight: Show total MST weight" << endl;
    cout << "  level_edges: Show MST edges by level" << endl;
    cout << "  euler_tour <vertex>: Show Euler Tour starting from vertex" << endl;
    cout << "  exit: Exit" << endl;

    string command;
    cin.ignore();
    while (true) {
        cout << "\nEnter command: ";
        getline(cin, command);
        if (command.empty()) continue;
        istringstream iss(command);
        string cmd;
        iss >> cmd;
        if (cmd == "exit") break;
        else if (cmd == "add_node") {
            char node;
            iss >> node;
            if (node < 'a' || node > 'z') {
                cout << "Invalid node: Use lowercase letters a-z" << endl;
                continue;
            }
            mst.add_node(node);
            cout << "Added node " << node << endl;
        } else if (cmd == "add_edge") {
            char u, v;
            double weight;
            iss >> u >> v >> weight;
            if (u < 'a' || u > 'z' || v < 'a' || v > 'z') {
                cout << "Invalid nodes: Use lowercase letters a-z" << endl;
                continue;
            }
            if (mst.add_edge(u, v, weight)) {
                cout << "Added edge (" << u << ", " << v << ") with weight " << weight << endl;
            } else {
                cout << "Edge already exists or invalid" << endl;
            }
        } else if (cmd == "remove_edge") {
            char u, v;
            int level;
            iss >> u >> v;
            if (u < 'a' || u > 'z' || v < 'a' || v > 'z') {
                cout << "Invalid nodes: Use lowercase letters a-z" << endl;
                continue;
            }
            if (mst.remove_edge(u, v, level)) {
                cout << "Removed edge (" << u << ", " << v << ") from level " << level << endl;
            } else {
                cout << "Edge does not exist" << endl;
            }
        } else if (cmd == "connected") {
            char u, v;
            iss >> u >> v;
            if (u < 'a' || u > 'z' || v < 'a' || v > 'z') {
                cout << "Invalid nodes: Use lowercase letters a-z" << endl;
                continue;
            }
            if (mst.same_component(u, v)) {
                cout << "Nodes " << u << " and " << v << " are in the same component" << endl;
            } else {
                cout << "Nodes " << u << " and " << v << " are not in the same component" << endl;
            }
        } else if (cmd == "mst_weight") {
            mst.get_mst_weight();
        } else if (cmd == "level_edges") {
            mst.display_level_edges();
        } else if (cmd == "euler_tour") {
            char vertex;
            iss >> vertex;
            if (vertex < 'a' || vertex > 'z') {
                cout << "Invalid vertex: Use lowercase letters a-z" << endl;
                continue;
            }
            vector<char> tour = mst.euler_tour(vertex);
            if (tour.empty()) {
                cout << "Vertex " << vertex << " not in MST or no tree edges" << endl;
            } else {
                cout << "Euler tour starting from " << vertex << ": ";
                for (char v : tour) cout << v << " ";
                cout << endl;
            }
        } else {
            cout << "Unknown command. Available commands: add_node, add_edge, remove_edge, connected, mst_weight, level_edges, euler_tour, exit" << endl;
        }
    }
}

int main() {
    run_interactive_demo();
    return 0;
}