"""Expression trees: parsing, pre-order tokenization and tree positional embedding."""
import json
import re

import networkx as nx
import torch

TREE_PE_DIM = 512


def parse_expression(expression, G=None, parent=None, label=""):
    """Parse 'NAND(X[0],AND(X[1],X[2]))' into a directed tree (edges point to children)."""
    if G is None:
        G = nx.DiGraph()

    # Remove outermost parentheses for easier parsing
    expression = expression.strip()
    if expression[0] == '(' and expression[-1] == ')':
        expression = expression[1:-1]

    match = re.match(r'(\w+)\((.*)\)', expression)
    if match:
        operation = match.group(1)
        operands = match.group(2)

        node = f"{label}_{operation}"
        G.add_node(node, label=operation)
        if parent:
            G.add_edge(parent, node)

        # Split the operands at top-level commas
        depth = 0
        parts = []
        start = 0
        for i, char in enumerate(operands):
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            elif char == ',' and depth == 0:
                parts.append(operands[start:i])
                start = i + 1
        parts.append(operands[start:])

        for i, part in enumerate(parts):
            parse_expression(part, G, node, label + str(i))
    else:
        # Leaf node (variable)
        variable = expression.strip()
        G.add_node(variable, label=variable)
        if parent:
            G.add_edge(parent, variable)

    return G


def visualize_tree(G):
    """Draw a tree (requires pygraphviz)."""
    import matplotlib.pyplot as plt
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    labels = nx.get_node_attributes(G, 'label')
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, labels=labels, with_labels=True, node_size=2000, node_color='lightblue', font_size=10, font_weight='bold')
    plt.show()


def save_graph(G, filename):
    with open(filename, 'w') as f:
        json.dump(nx.node_link_data(G), f)


def load_graph(filename):
    with open(filename, 'r') as f:
        return nx.node_link_graph(json.load(f))


def preorder_traversal(G):
    """Token sequence of the tree in pre-order (children visited in sorted order)."""
    start = [n for n, d in G.in_degree() if d == 0][0]
    traversal_result = []

    def dfs(node):
        traversal_result.append(G.nodes[node]['label'])
        for neighbor in sorted(G.successors(node)):
            dfs(neighbor)

    dfs(start)
    return traversal_result


def tree_to_text(G):
    """Pre-order tokens with variable indices dropped: 'NAND X AND X X'."""
    return re.sub(r'X\[\d+\]', 'X', ' '.join(preorder_traversal(G)))


def encode_tree(G):
    """Tree positional code per node, in pre-order.

    The root is all zeros; each step down shifts the parent code right by two
    and prepends '10' for the first child or '01' for the second.
    """
    encoding = {}

    def encode(node, current_code):
        children = sorted(G.successors(node))
        for i, child in enumerate(children):
            if len(children) == 1 or i == 0:
                new_code = '10' + current_code[:-2]
            else:
                new_code = '01' + current_code[:-2]
            encoding[child] = new_code
            encode(child, new_code)

    root = None
    for node in G.nodes:
        if G.in_degree(node) == 0:
            root = node
    root_code = '0' * TREE_PE_DIM
    encoding[root] = root_code
    encode(root, root_code)

    return [list(map(int, code)) for code in encoding.values()]


def tree_positional_embedding(G):
    return torch.tensor(encode_tree(G), dtype=torch.float)
