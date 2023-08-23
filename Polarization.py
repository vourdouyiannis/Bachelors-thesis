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

# comments = [
#     [
#         {
#             "id": "comment1",
#             "author": "User1",
#             "parent_id": "",
#             "content": "I strongly believe in this.",
#             "sentiment": 0.8
#         },
#         {
#             "id": "comment2",
#             "author": "User2",
#             "parent_id": "comment1",
#             "content": "Absolutely! It's the best option.",
#             "sentiment": 0.7
#         },
#         {
#             "id": "comment3",
#             "author": "User3",
#             "parent_id": "comment1",
#             "content": "This is completely wrong. We should never do that.",
#             "sentiment": -0.6
#         },
#         {
#             "id": "comment4",
#             "author": "User4",
#             "parent_id": "comment2",
#             "content": "I strongly disagree. We need to explore other alternatives.",
#             "sentiment": -0.7
#         },
#         {
#             "id": "comment5",
#             "author": "User5",
#             "parent_id": "comment3",
#             "content": "You're all mistaken! There's a better way.",
#             "sentiment": -0.8
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
#             "sentiment": 0.3
#         },
#         {
#             "id": "comment2",
#             "author": "User2",
#             "parent_id": "comment1",
#             "content": "It's worth considering.",
#             "sentiment": 0.2
#         },
#         {
#             "id": "comment3",
#             "author": "User3",
#             "parent_id": "comment1",
#             "content": "I'm not sure about this idea.",
#             "sentiment": -0.1
#         },
#         {
#             "id": "comment4",
#             "author": "User4",
#             "parent_id": "comment2",
#             "content": "I don't have a strong opinion on this matter.",
#             "sentiment": 0.0
#         },
#         {
#             "id": "comment5",
#             "author": "User5",
#             "parent_id": "comment3",
#             "content": "Can you provide more details?",
#             "sentiment": 0.1
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
    # Determine if the comment is positive or negative based on the sentiment score
    # No score
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
        sentiment = comment['sentiment']
        G.add_node(author, label=sentiment)
        # polarity_dict[author] = float(sentiment) if sentiment != '' else 0.0

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

    # Use the get_sentiment function to get the symbol we want to print for each node ("0", "+", "-")
    node_labels = {node: get_sentiment(merged_graph.nodes[node].get('label')) for node in merged_graph.nodes()}
    nx.draw_networkx_labels(merged_graph, pos=label_pos, labels=node_labels)

    return merged_graph


def calculate_polarization_score(G, polarity_dict, num_walks, walk_length):
    """
    Calculates the polarization score of a graph based on the sentiment values
    of its nodes and the probabilities obtained from random walks.

    Parameters:
        G: The input graph.
        polarity_dict: Dictionary with nodes as keys and sentiment values as values.
        num_walks: Number of random walks to perform from each node.
        walk_length: Length of each random walk.

    Returns:
        polarization_score: Computed polarization score.
    """

    # Compute walk probabilities, which give the probability of ending up in a
    # node with a certain sentiment value after a random walk.
    walk_probabilities = random_walk_probability(G, num_walks, walk_length)

    total_difference = 0.0
    num_nodes = 0

    # For each node, compute the expected sentiment value using the walk probabilities
    for node, sentiments in walk_probabilities.items():
        # Calculate the expected sentiment value based on the probabilities.
        # It's a weighted average where probabilities serve as weights.
        expected_value = sum([sentiments.get(str(s), 0) * s for s in [-1, 0, 1]])

        # Fetch the actual sentiment value of the node from polarity_dict
        actual_value = polarity_dict.get(node, 0)

        # Add up the absolute difference between expected and actual values
        total_difference += abs(expected_value - actual_value)

        # Increment the count of nodes processed
        num_nodes += 1

    # Average the total difference to get the final polarization score
    polarization_score = total_difference / num_nodes if num_nodes else 0

    return round(polarization_score, 3)


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

        for i in range(num_walks):
            current_node = node
            for j in range(walk_length):
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

    # Loop through each node in the graph and extract the 'label' attribute which is the sentiment analysis - "sentiment"
    for node in G.nodes():
        sentiment = G.nodes[node].get('label')

        # If 'label' exists and is not an empty string, attempt to convert it to a floating-point number
        if sentiment is not None and sentiment != "":
            try:
                polarity_values.append(float(sentiment))
                node_labels[node] = sentiment
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
def save_to_file2(ctr, contr):
    filename = ""
    if contr == 0:
        filename = "graphs/non_controversial/Posts{}_and_{}_graph.png".format(ctr, ctr + 1)
    if contr == 1:
        filename = "graphs/controversial/Posts{}_and_{}_graph.png".format(ctr, ctr + 1)
    plt.savefig(filename)
    plt.clf()


def results_for_one_graph(counter, num_walks, walk_length, kmeans):
    """
    The reason that the graph is drawn here
    is that we don't want to be drawn
    when we draw the graphs with the 2 posts
    """

    for i in range(len(comments)):
        polarity_dict = {}  # A dictionary that keeps the sentiment values
        G = create_graph(comments[i])

        # Get the polarity_dict:
        for comment in comments[i]:
            polarity_dict[comment['author']] = float(comment['sentiment']) if comment['sentiment'] != '' else 0.0

        # Draw the graph
        pos = nx.spring_layout(G, k=0.05)
        nx.draw(G, pos, node_size=2, node_color='blue', edge_color='gray')

        # Add labels in the graph
        label_offset = 0.04  # A small offset to position the labels perfectly
        label_pos = {k: (v[0], v[1] + label_offset) for k, v in pos.items()}

        # Use the get_sentiment function to get the symbol we want to print for each node ("0", "+", "-")
        node_labels = {node: get_sentiment(G.nodes[node].get('label')) for node in G.nodes()}
        nx.draw_networkx_labels(G, pos=label_pos, labels=node_labels)

        save_to_file(counter)

        polarization_score = calculate_polarization_score(G, polarity_dict, num_walks, walk_length)
        print(f"Polarization score for post {counter}: {polarization_score}")

        # Detect groups based on agreement using SCG
        groups = SCG(G, kmeans)

        # Print the nodes in each group
        for j, group in enumerate(groups):
            print(f"Group {j + 1}: {group.nodes()}")

        counter += 1


def results_for_two_graphs(counter, controversial, num_walks, walk_length, kmeans):
    for i in range(len(comments)):
        polarity_dict = {}
        if i < len(comments) - 1 and i != 9 and controversial == 0:
            G2 = create_graph2(comments[i], comments[i + 1])

            # Get the polarity_dict:
            for comment in comments[i]:
                polarity_dict[comment['author']] = float(comment['sentiment']) if comment['sentiment'] != '' else 0.0
            for comment in comments[i + 1]:
                polarity_dict[comment['author']] = float(comment['sentiment']) if comment['sentiment'] != '' else 0.0

            save_to_file2(counter, controversial)

            polarization_score = calculate_polarization_score(G2, polarity_dict, num_walks, walk_length)
            print("Polarization score for posts {} and {}: {}".format(counter, counter + 1, polarization_score))

            # Detect groups based on agreement using SCG
            groups = SCG(G2, kmeans)

            # Print the nodes in each group
            for j, group in enumerate(groups):
                print(f"Group {j + 1}: {group.nodes()}")

        if i < len(comments) - 5 and controversial == 1 and i != 9:
            G2 = create_graph2(comments[i], comments[i + 5])

            # Get the polarity_dict:
            for comment in comments[i]:
                polarity_dict[comment['author']] = float(comment['sentiment']) if comment['sentiment'] != '' else 0.0
            for comment in comments[i + 5]:
                polarity_dict[comment['author']] = float(comment['sentiment']) if comment['sentiment'] != '' else 0.0

            save_to_file2(counter, controversial)

            polarization_score = calculate_polarization_score(G2, polarity_dict, num_walks, walk_length)
            print("Polarization score for posts {} and {}: {}".format(counter, counter + 5, polarization_score))

            # Detect groups based on agreement using SCG
            groups = SCG(G2, kmeans)

            # Print the nodes in each group
            for j, group in enumerate(groups):
                print(f"Group {j + 1}: {group.nodes()}")

        if i == 8:
            controversial = 1

        counter += 1


# Required fields to run the processes
counter = 1  # Counts the post we are currently processing
controversial = 0  # controversial: 0:no, 1:yes

num_walks = 100  # Number of random walks to perform from each node
walk_length = 10  # Length of each random walk
k = 3  # The number of the groups I want to detect

# Run the processes
results_for_one_graph(counter, num_walks, walk_length, k)
results_for_two_graphs(counter, controversial, num_walks, walk_length, k)
