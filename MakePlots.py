# VOURDOUGIANNIS DIMITRIOS 4326 #

import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Read data from JSON file
with open('resources/comments.json', 'r') as f:
    comments = json.load(f)


def remove_nodes(G, post):
    # Removing self-loop edges
    G.remove_edges_from(list(nx.selfloop_edges(G)))

    # Get the nodes without edges
    # Remove the isolated nodes
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    # Specify the node to keep
    # Then find the nodes that are not connected to the node to keep
    # And remove the isolated nodes
    node_to_keep = post[0]['author']
    isolated_nodes = [n for n in G.nodes() if not nx.has_path(G, node_to_keep, n)]
    G.remove_nodes_from(isolated_nodes)


def create_edges(G, post):
    for i in range(len(post)):
        for j in range(len(post)):
            if post[i]['parent_id'] == post[j]['id']:
                G.add_edge(post[i]['author'], post[j]['author'])


def get_sentiment(sentiment):
    # Determine if the comment is positive or negative based on the polarity score
    # No polarity
    if sentiment == "":
        return "0"
    # Positive comment
    if float(sentiment) > 0:
        return "+"
    # Negative comment
    elif float(sentiment) < 0:
        return "-"
    # Neutral comment
    else:
        return "0"


def create_graph(post):
    # Create an empty graph
    G = nx.Graph()

    # Add nodes to the graph
    for comment in post:
        author = comment['author']
        polarity = comment['polarity']
        G.add_node(comment['author'], label=get_sentiment(comment['polarity']))
        if polarity == '':
            polarity_dict[author] = 0.0
        else:
            polarity_dict[author] = polarity
    #print(polarity_dict)

    # Add edges to the graph
    create_edges(G, post)

    # Remove the nodes we don't need or causing a problem
    remove_nodes(G, post)

    return G


# Create a graph for two different posts
def create_graph2(post1, post2):
    graph1 = create_graph(post1)
    graph2 = create_graph(post2)

    nodes1 = graph1.nodes()
    nodes2 = graph2.nodes()
    common_nodes = nodes1 & nodes2

    # Merge the two graphs
    merged_graph = nx.compose(graph1, graph2)

    # Set nodes color
    node_color = []
    for node in merged_graph.nodes():
        if node in common_nodes:
            node_color.append('purple')
        elif node in nodes1:
            node_color.append('blue')
        else:
            node_color.append('red')

    # Set nodes position
    pos = nx.spring_layout(merged_graph, k=0.01)

    # Draw the merged graph with different colors
    nx.draw(merged_graph, pos=pos, node_size=10, node_color=node_color, edge_color='gray')

    # Add labels in the graph
    label_offset = 0.04  # A small offset to position the labels perfectly
    label_pos = {k: (v[0], v[1] + label_offset) for k, v in pos.items()}
    nx.draw_networkx_labels(merged_graph, pos=label_pos, labels=nx.get_node_attributes(merged_graph, 'label'))

    return merged_graph


def calculate_polarization_score(G, polarity_dict):
    # Get the polarity values for each node
    node_values = [float(polarity_dict[node]) if node in polarity_dict else 0.0 for node in G.nodes()]

    # Compute the initial and final distributions
    initial_distribution = np.array(node_values) / sum(node_values)
    final_distribution = np.zeros(len(G.nodes()))

    # Create a dictionary to store the indices of the nodes
    node_indices = {node: idx for idx, node in enumerate(G.nodes())}

    # Perform the random walk
    alpha = 0.85  # Damping factor
    for i in range(100):
        for node in G.nodes():
            neighbors = [n for n in G.neighbors(node) if G.has_edge(node, n)]
            contribution = sum(initial_distribution[node_indices[neighbor]] for neighbor in neighbors) if neighbors else 0
            final_distribution[node_indices[node]] += alpha * (contribution / len(neighbors)) if neighbors else 0

        final_distribution += (1 - alpha) * (initial_distribution / len(G.nodes()))
        initial_distribution = final_distribution.copy()
        final_distribution = np.zeros(len(G.nodes()))

    # Calculate the polarization score
    polarization_score = np.sum(np.abs(initial_distribution - 1 / len(G.nodes())))

    return polarization_score


def save_to_file(ctr):
    # Save each graph to a file
    filename = "graphs/all_posts/Post{}_graph.png".format(ctr)
    plt.savefig(filename)
    plt.clf()


# Save each graph to a file
def save_to_file2(ctr, subr, contr):
    filename = ""
    if contr == 0:
        if subr == 0:
            filename = "graphs/non_controversial/different_subreddit/Posts{}_and_{}_graph.png".format(ctr, ctr + 1)
        if subr == 1:
            filename = "graphs/non_controversial/same_subreddit/Posts{}_and_{}_graph.png".format(ctr, ctr + 1)
    if contr == 1:
        if subr == 0:
            filename = "graphs/controversial/different_subreddit/Posts{}_and_{}_graph.png".format(ctr, ctr + 1)
        if subr == 1:
            filename = "graphs/controversial/same_subreddit/Posts{}_and_{}_graph.png".format(ctr, ctr + 1)
    plt.savefig(filename)
    plt.clf()


# counter = 1
# for post in comments:
#     polarity_dict = {}
#     G = create_graph(post)
#
#     # Draw the graph
#     pos = nx.spring_layout(G, k=0.05, scale=0.1, iterations=50)
#     nx.draw(G, pos, node_size=2, node_color='blue', edge_color='gray')
#
#     # Add labels in the graph
#     label_offset = 0.003  # A small offset to position the labels perfectly
#     label_pos = {k: (v[0], v[1] + label_offset) for k, v in pos.items()}
#     nx.draw_networkx_labels(G, pos=label_pos, labels=nx.get_node_attributes(G, 'label'))
#
#     save_to_file(counter)
#     counter += 1

counter = 1
controversial = 0
subreddit = 1
# controversial: 0:no, 1:yes
# subreddit: 0:different, 1:same

for i in range(len(comments)):
    polarity_dict = {}
    G = create_graph(comments[i])

    # Draw the graph
    pos = nx.spring_layout(G, k=0.05, scale=0.1, iterations=50)
    nx.draw(G, pos, node_size=2, node_color='blue', edge_color='gray')

    # Add labels in the graph
    label_offset = 0.003  # A small offset to position the labels perfectly
    label_pos = {k: (v[0], v[1] + label_offset) for k, v in pos.items()}
    nx.draw_networkx_labels(G, pos=label_pos, labels=nx.get_node_attributes(G, 'label'))

    polarization_score = calculate_polarization_score(G, polarity_dict)
    print(f"Polarization score for post {counter}: {polarization_score}")

    save_to_file(counter)

    counter += 1
    if counter == 10:
        controversial = 1
    # for number2 in polarity_dict:
    #     print(polarity_dict[number2])

counter = 1
for i in range(len(comments)):
    polarity_dict = {}
    if i < len(comments) - 1:
        G2 = create_graph2(comments[i], comments[i + 1])

        polarization_score = calculate_polarization_score(G2, polarity_dict)
        print("Polarization score for posts {} and {}: {}".format(counter, counter + 1, polarization_score))

        save_to_file2(counter, subreddit, controversial)

    counter += 1
    # for number2 in polarity_dict:
    #     print(polarity_dict[number2])
