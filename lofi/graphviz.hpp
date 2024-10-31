#pragma once
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <set>
#include <sstream>
#include <tuple>

#include <lofi/engine.hpp>

template <typename T> std::string context_to_graphviz(const std::shared_ptr<Context<T>> &ctx) {
    std::stringstream node_label;
    node_label << "{";
    if (!ctx->label.empty()) {
        node_label << ctx->label;
    } else {
        node_label << ctx->data.id();
    }

    node_label << " | " << ctx->shape() << "}";
    std::stringstream ss;
    ss << "\"" << ctx.get() << "\" [label=\"" << node_label.str() << "\", shape=record];\n";

    if (!ctx->op.empty()) {
        ss << "\"" << ctx.get() << "_op\" [label=\"" << ctx->op << "\"];\n";
        ss << "\"" << ctx.get() << "_op\" -> \"" << ctx.get() << "\";\n";
    }

    return ss.str();
}

template <typename T> auto trace(Matrix<T> &root) {
    using ctx_ptr_t = std::shared_ptr<Context<T>>;

    std::unordered_set<ctx_ptr_t> nodes;
    std::vector<std::pair<ctx_ptr_t, ctx_ptr_t>> edges;

    std::function<void(ctx_ptr_t &)> build = [&](ctx_ptr_t &node) {
        if (!nodes.contains(node)) {
            nodes.insert(node);
            for (auto &child : node->prev) {
                edges.push_back({child, node});
                build(child);
            }
        }
    };

    build(root.ctx_);
    return std::make_pair(nodes, edges);
}

template <typename T> void draw_dot(Matrix<T> &root, const std::string &filename, const std::string &rankdir = "LR") {
    if (rankdir != "LR" && rankdir != "TB") {
        std::cerr << "Invalid rankdir. Use 'LR' or 'TB'." << std::endl;
        return;
    }

    auto [nodes, edges] = trace(root);

    // Create the .dot file
    std::ofstream file(filename);

    file << "digraph G {\n";
    file << "rankdir=" << rankdir << ";\n";

    // Add nodes to the dot file
    for (auto &n : nodes) {
        file << context_to_graphviz(n);
    }

    // Add edges to the dot file
    for (const auto &edge : edges) {
        const auto &n1 = edge.first;
        const auto &n2 = edge.second;
        file << "\"" << n1.get() << "\" -> \"" << n2.get() << "_op\";\n";
    }

    file << "}\n";
    file.close();

    std::cout << "Graphviz .dot file generated: " << filename << std::endl;
}

void generate_svg_from_dot(const std::string &dot_filename, const std::string &out_filename) {
    // Construct the command to convert .dot to .svg using Graphviz's dot tool
    std::string command = "dot -Tsvg " + dot_filename + " -o " + out_filename;

    // Execute the command
    int result = system(command.c_str());

    // Check if the command succeeded
    if (result == 0) {
        std::cout << "SVG image successfully created: " << out_filename << std::endl;
    } else {
        std::cerr << "Failed to create SVG image. Make sure Graphviz is installed and 'dot' is in your PATH."
                  << std::endl;
    }
}
