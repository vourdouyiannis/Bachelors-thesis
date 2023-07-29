# VOURDOUGIANNIS DIMITRIOS 4326 #

import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.cluster import KMeans

# Read data from JSON file
with open('resources/comments.json', 'r') as f:
    comments = json.load(f)

# High polarization score example input data:
#
# comments = [
#     [
#         {
#             "id": "comment1",
#             "author": "User1",
#             "parent_id": "",
#             "content": "I strongly believe in this.",
#             "polarity": 0.8
#         },
#         {
#             "id": "comment2",
#             "author": "User2",
#             "parent_id": "comment1",
#             "content": "Absolutely! It's the best option.",
#             "polarity": 0.7
#         },
#         {
#             "id": "comment3",
#             "author": "User3",
#             "parent_id": "comment1",
#             "content": "This is completely wrong. We should never do that.",
#             "polarity": -0.6
#         },
#         {
#             "id": "comment4",
#             "author": "User4",
#             "parent_id": "comment2",
#             "content": "I strongly disagree. We need to explore other alternatives.",
#             "polarity": -0.7
#         },
#         {
#             "id": "comment5",
#             "author": "User5",
#             "parent_id": "comment3",
#             "content": "You're all mistaken! There's a better way.",
#             "polarity": -0.8
#         }
#     ]
# ]


# Low polarization score example input data:

# comments = [
#     [
#         {
#             "id": "comment1",
#             "author": "User1",
#             "parent_id": "",
#             "content": "I think this could work.",
#             "polarity": 0.3
#         },
#         {
#             "id": "comment2",
#             "author": "User2",
#             "parent_id": "comment1",
#             "content": "It's worth considering.",
#             "polarity": 0.2
#         },
#         {
#             "id": "comment3",
#             "author": "User3",
#             "parent_id": "comment1",
#             "content": "I'm not sure about this idea.",
#             "polarity": -0.1
#         },
#         {
#             "id": "comment4",
#             "author": "User4",
#             "parent_id": "comment2",
#             "content": "I don't have a strong opinion on this matter.",
#             "polarity": 0.0
#         },
#         {
#             "id": "comment5",
#             "author": "User5",
#             "parent_id": "comment3",
#             "content": "Can you provide more details?",
#             "polarity": 0.1
#         }
#     ]
# ]


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
        G.add_node(author, label=polarity)
        polarity_dict[author] = float(polarity) if polarity != '' else 0.0

    # Add edges to the graph
    create_edges(G, post)

    # Remove the nodes we don't need or causing a problem
    remove_nodes(G, post)

    # Draw the graph
    pos = nx.spring_layout(G, k=0.05)
    nx.draw(G, pos, node_size=2, node_color='blue', edge_color='gray')

    # Add labels in the graph
    label_offset = 0.04  # A small offset to position the labels perfectly
    label_pos = {k: (v[0], v[1] + label_offset) for k, v in pos.items()}

    # Use the get_sentiment function to get the symbol we want to print for each node ("0", "+", "-")
    node_labels = {node: get_sentiment(G.nodes[node].get('label')) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos=label_pos, labels=node_labels)

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

    # Use the get_sentiment function to get the symbol we want to print for each node ("0", "+", "-")
    node_labels = {node: get_sentiment(merged_graph.nodes[node].get('label')) for node in merged_graph.nodes()}
    nx.draw_networkx_labels(merged_graph, pos=label_pos, labels=node_labels)

    return merged_graph


# def calculate_polarization_score(G, polarity_dict):
#     node_values = [float(polarity_dict[node]) if node in polarity_dict else 0.0 for node in G.nodes()]
#     sum_node_values = sum(node_values)
#
#     if sum_node_values != 0:
#         initial_distribution = np.array(node_values) / sum_node_values
#         alpha = 0.85
#         final_distribution = nx.pagerank(G, alpha=alpha, personalization=None, max_iter=1000, tol=1e-06)
#         initial_polarity_values = np.array([float(polarity_dict[node]) if node in polarity_dict else 0.0 for node in final_distribution.keys()])
#         polarization_score = np.sum(np.abs(initial_distribution - initial_polarity_values))
#         normalization_factor = len(G.nodes()) - 1
#         polarization_score /= normalization_factor
#
#         # Normalize the polarization score to be within the range of 0 to 1
#         polarization_score = min(max(polarization_score, 0), 1)
#     else:
#         polarization_score = 0.0  # Set polarization score to 0 if sum_node_values is 0
#
#     return polarization_score


def calculate_polarization_score(G, polarity_dict, num_walks, walk_length):
    # Step 1: Calculate the initial distribution of polarity values for nodes in the graph
    node_values = [float(polarity_dict[node]) if node in polarity_dict else 0.0 for node in G.nodes()]
    sum_node_values = sum(node_values)

    # Step 2: Calculate the polarization score using random walk
    if sum_node_values != 0:
        # Calculate the initial distribution by dividing each node's polarity value by the sum of all polarity values
        initial_distribution = np.array(node_values) / sum_node_values

        # Perform random walks from each node to estimate the probability of reaching a node with a specific polarity
        walk_probabilities = random_walk_probability(G, num_walks=num_walks, walk_length=walk_length)

        # Calculate the initial polarity values for nodes in the same order as the walk probabilities keys
        initial_polarity_values = np.array(
            [float(polarity_dict[node]) if node in polarity_dict else 0.0 for node in walk_probabilities.keys()])

        # Calculate the polarization score as the absolute difference between initial distribution and walk probabilities
        polarization_score = np.sum(np.abs(initial_distribution - initial_polarity_values))

        # Calculate a normalization factor based on the number of nodes in the graph
        normalization_factor = len(G.nodes()) - 1

        # Normalize the polarization score to be within the range of 0 to 1
        polarization_score /= normalization_factor
        polarization_score = min(max(polarization_score, 0), 1)
    else:
        # If the sum of node polarity values is 0, set the polarization score to 0
        polarization_score = 0.0

    # Step 3: Return the calculated polarization score
    return polarization_score


def random_walk_probability(G, num_walks, walk_length):
    """
    Perform random walks on the graph G and calculate the probability of reaching each node with a specific polarity.

    Parameters:
        G (nx.Graph): The input graph.
        num_walks (int): Number of random walks to perform from each node (default: 100).
        walk_length (int): Length of each random walk (default: 10).

    Returns:
        dict: A dictionary with nodes as keys and their corresponding probability of reaching with a specific polarity as values.
    """
    walk_probabilities = {}
    for node in G.nodes():
        walk_counts = {polarity: 0 for polarity in set(G.nodes[node].get('label') for node in G.nodes()) if
                       polarity != ''}

        for _ in range(num_walks):
            current_node = node
            for _ in range(walk_length):
                neighbors = list(G.neighbors(current_node))
                if len(neighbors) == 0:
                    break
                current_node = random.choice(neighbors)
                current_polarity = G.nodes[current_node].get('label')
                if current_polarity is not None and current_polarity != '':
                    walk_counts[current_polarity] += 1

        total_walks = sum(walk_counts.values())
        walk_probabilities[node] = {polarity: count / total_walks for polarity, count in walk_counts.items()}

    return walk_probabilities


def SCG(G, k):
    # Get the polarity values of nodes and their labels
    polarity_values = []
    node_labels = {}

    # Loop through each node in the graph and extract the 'label' attribute which is the sentiment analysis - "polarity"
    for node in G.nodes():
        polarity = G.nodes[node].get('label')

        # If 'label' exists and is not an empty string, attempt to convert it to a floating-point number
        if polarity is not None and polarity != "":
            try:
                polarity_values.append(float(polarity))
                node_labels[node] = polarity
            except ValueError:
                # If the conversion fails, skip this node and continue to the next one
                continue

    # Apply k-means clustering
    if len(polarity_values) < k:
        return []  # If there are not enough distinct polarity values to create 'k' clusters, return an empty list

    # Use k-means clustering to group polarity values into 'k' clusters
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(np.array(polarity_values).reshape(-1, 1))
    cluster_labels = kmeans.labels_

    # Create a group for each cluster
    groups = {}

    # Iterate through the nodes and their labels and group nodes with the same cluster label into subgraphs
    for i, (node, label) in enumerate(node_labels.items()):
        group_label = cluster_labels[i]  # Get the cluster label for the current node
        group_name = f"group{group_label}"

        if group_name not in groups:
            # If the group doesn't exist, create a new subgraph for this cluster
            groups[group_name] = nx.Graph()

        # Add the node to the appropriate subgraph with its label
        groups[group_name].add_node(node, label=label)

    # Return the list of groups, each representing one cluster
    return list(groups.values())

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
controversial = 0  # controversial: 0:no, 1:yes
subreddit = 1  # subreddit: 0:different, 1:same

num_walks = 100  # Number of random walks to perform from each node
walk_length = 10  # Length of each random walk
k = 3  # The number of the groups I want to detect

for i in range(len(comments)):
    polarity_dict = {}
    G = create_graph(comments[i])

    polarization_score = calculate_polarization_score(G, polarity_dict, num_walks, walk_length)
    print(f"Polarization score for post {counter}: {polarization_score}")

    # Detect groups based on agreement using SCG
    groups = SCG(G, k)

    # Print the nodes in each group
    for i, group in enumerate(groups):
        print(f"Group {i + 1}: {group.nodes()}")

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

        polarization_score = calculate_polarization_score(G2, polarity_dict, num_walks, walk_length)
        print("Polarization score for posts {} and {}: {}".format(counter, counter + 1, polarization_score))

        # Detect groups based on agreement using SCG
        groups = SCG(G2, k)

        # Print the nodes in each group
        for i, group in enumerate(groups):
            print(f"Group {i + 1}: {group.nodes()}")

        save_to_file2(counter, subreddit, controversial)

    counter += 1
    # for number2 in polarity_dict:
    #     print(polarity_dict[number2])
