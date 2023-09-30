# VOURDOUGIANNIS DIMITRIOS 4326 #

import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import rbf_kernel

# Read data from JSON files
with open('resources/comments_non_controversial.json', 'r') as f:
    comments_non_controversial = json.load(f)

with open('resources/comments_controversial.json', 'r') as f:
    comments_controversial = json.load(f)

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
#
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

    # These dictionaries will help to calculate the mean sentiment score for nodes
    # that appear multiple times
    accumulated_sentiments = {}  # To accumulate sentiments of nodes
    node_counts = {}  # To count occurrences of nodes

    # Add nodes to the graph
    for comment in post:
        author = comment['author']
        sentiment = float(comment['sentiment']) if comment['sentiment'] != '' else 0.0

        if author not in G:
            G.add_node(author, label=sentiment)
            accumulated_sentiments[author] = sentiment
            node_counts[author] = 1
        else:
            # Update the accumulated sentiment and count for the node
            accumulated_sentiments[author] += sentiment
            node_counts[author] += 1

            # Compute the mean sentiment and update the node's sentiment in the graph
            mean_sentiment = accumulated_sentiments[author] / node_counts[author]
            G.nodes[author]['label'] = mean_sentiment

    # An enas xrhsths exei sxoliasei panw apo mia fora se ena grafhma h kai se duo mazi tote upologizetai o mesos oros twn sxoliwn
    # kai mpainei sto grafhma o mesos oros aftos /TODO DELETE

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

    if not common_nodes:
        return 0
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
        G: The input graph.
        num_walks: Number of random walks to perform from each node (default: 100).
        walk_length: Length of each random walk (default: 10).

    Returns:
        dict: A dictionary with nodes as keys and their corresponding probability of reaching a node as values.
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


def compute_polarization_score_from_edges(G):
    total_difference = 0.0
    for node in G.nodes():
        actual_sentiment = G.nodes[node]['label']
        expected_sentiment = compute_expected_sentiment(node, G)
        total_difference += abs(actual_sentiment - expected_sentiment)

    # Normalize the polarization score to [0,1]
    polarization_score = total_difference / len(G.nodes())
    return min(round(polarization_score, 3), 1)  # Ensure the score doesn't exceed 1


def compute_expected_sentiment(node, G):
    neighbors = list(G.neighbors(node))
    if not neighbors:  # If node has no neighbors
        return G.nodes[node]['label']

    weighted_sum = 0.0
    total_weight = 0.0
    for neighbor in neighbors:
        abs_diff = abs(G.nodes[node]['label'] - G.nodes[neighbor]['label'])
        if abs_diff > 1:
            edge_weight = 1
        else:
            edge_weight = 0

        G[node][neighbor]['weight'] = edge_weight

        weighted_sum += edge_weight * G.nodes[neighbor]['label']
        total_weight += abs(edge_weight)

    # Normalize the expected sentiment, so it's bounded between -1 and 1
    return weighted_sum / total_weight if total_weight != 0 else 0


def SCG(G, K):
    # Remove isolated nodes
    G.remove_nodes_from(list(nx.isolates(G)))

    # Use adjacency matrix
    adj_matrix = nx.adjacency_matrix(G).todense()

    # Convert the adjacency matrix to a similarity matrix using Gaussian kernel
    similarity_matrix = rbf_kernel(adj_matrix)

    # Check for NaN values
    if np.any(np.isnan(similarity_matrix)):
        print("Similarity matrix contains NaN values!")

    # Check if the number of nodes is less than the number of clusters you want
    if len(G.nodes()) <= K:
        print(f"Graph has {len(G.nodes())} nodes, which is less than {K} clusters. Ignoring this graph.")
        return []

    try:
        # Apply Spectral Clustering
        clustering = SpectralClustering(n_clusters=K, affinity='precomputed', random_state=42).fit(similarity_matrix)
    except RuntimeWarning:
        print(
            f"Failed to perform spectral clustering on graph with {len(G.nodes())} nodes and {K} clusters. Ignoring this graph.")
        return []
    return clustering.labels_


def plot_graph_with_clusters(G, clusters):
    nodes_list = list(G.nodes())

    # Ensure clusters has an entry for each node
    if len(clusters) != len(nodes_list):
        print("Mismatch between nodes and clusters. Skipping plot.")
        return

    color_map = [clusters[i] for i, node in enumerate(nodes_list)]
    pos = nx.spring_layout(G, k=0.05)
    nx.draw(G, pos, node_size=10, node_color=color_map, edge_color='gray')
    return G


def save_to_file(ctr):
    # Save each graph to a file
    filename = "graphs/all_posts/Post{}_graph.png".format(ctr)
    plt.savefig(filename)
    plt.clf()


def save_to_file2(ctr, contr):
    # Save each graph to a file
    filename = ""
    if contr == 0:
        filename = "graphs/non_controversial/Posts{}_and_{}_graph.png".format(ctr, ctr + 1)
    if contr == 1:
        filename = "graphs/controversial/Posts{}_and_{}_graph.png".format(ctr, ctr + 1)
    plt.savefig(filename)
    plt.clf()


def save_to_group_file(ctr):
    # Save each graph to a file
    filename = "graphs/all_posts_grouped/Post{}_graph.png".format(ctr)
    plt.savefig(filename)
    plt.clf()


def results_for_one_graph(num_walks, walk_length, k):
    """
    The reason that the graph is drawn here
    is that we don't want to be drawn
    when we draw the graphs with the 2 posts
    """

    for i in range(len(comments_non_controversial)):
        polarity_dict = {}  # A dictionary that keeps the sentiment values
        G = create_graph(comments_non_controversial[i])
        # Get the polarity_dict:
        for comment in comments_non_controversial[i]:
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

        save_to_file(i)

        polarization_score = calculate_polarization_score(G, polarity_dict, num_walks, walk_length)
        print(f"Polarization score - Random walk - for post {i+1}: {polarization_score}")

        # Detect groups based on agreement using SCG
        # groups = SCG(G, kmeans)

        polarization_score2 = compute_polarization_score_from_edges(G)
        print(f"Polarization score - from edges - for post {i+1}: {polarization_score2}")
        print("---------------------------------------------")

        # Detect groups and draw them
        clusters = SCG(G, k)
        G = plot_graph_with_clusters(G, clusters)

        save_to_group_file(i+1)
        # Print the nodes in each group
        # for j, group in enumerate(groups):
        #     print(f"Group {j + 1}: {group.nodes()}")


    for i in range(len(comments_controversial)):
        polarity_dict = {}  # A dictionary that keeps the sentiment values
        G = create_graph(comments_controversial[i])
        # Get the polarity_dict:
        for comment in comments_controversial[i]:
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

        save_to_file(len(comments_non_controversial)+i)

        polarization_score = calculate_polarization_score(G, polarity_dict, num_walks, walk_length)
        print(f"Polarization score - Random walk - for post {len(comments_non_controversial) + i + 1}: {polarization_score}")

        # Detect groups based on agreement using SCG
        # groups = SCG(G, kmeans)

        # clusters = SCG(G, k)
        # plot_graph_with_clusters(G, clusters)

        polarization_score2 = compute_polarization_score_from_edges(G)
        print(f"Polarization score - from edges - for post {len(comments_non_controversial) + i + 1}: {polarization_score2}")
        print("---------------------------------------------")
        # Print the nodes in each group
        # for j, group in enumerate(groups):
        #     print(f"Group {j + 1}: {group.nodes()}")


def results_for_two_graphs(num_walks, walk_length, k):
    counter = 0
    for i in range(len(comments_non_controversial) - 1):
        polarity_dict = {}
        G2 = create_graph2(comments_non_controversial[i], comments_non_controversial[i + 1])

        if G2 == 0 or G2 is None:
            counter += 1
            continue
        else:
            # Get the polarity_dict:
            for comment in comments_non_controversial[i]:
                polarity_dict[comment['author']] = float(comment['sentiment']) if comment['sentiment'] != '' else 0.0
            for comment in comments_non_controversial[i + 1]:
                polarity_dict[comment['author']] = float(comment['sentiment']) if comment['sentiment'] != '' else 0.0

            save_to_file2(i+1, 0)

            polarization_score = calculate_polarization_score(G2, polarity_dict, num_walks, walk_length)
            print("Non-Controversial: Polarization score - Random walk - for posts {} and {}: {}".format(i+1, i + 2, polarization_score))

            # Detect groups based on agreement using SCG
            # groups = SCG(G2, kmeans)

            polarization_score2 = compute_polarization_score_from_edges(G2)
            print("Non-Controversial: Polarization score - From edges - for posts {} and {}: {}".format(i+1, i + 2, polarization_score2))

            print("---------------------------------------------")
            # Print the nodes in each group
            # for j, group in enumerate(groups):
            #     print(f"Group {j + 1}: {group.nodes()}")

    print("#############################################")
    for i in range(len(comments_controversial) - 1):
        counter = 0
        polarity_dict = {}
        G2 = create_graph2(comments_controversial[i], comments_controversial[i + 1])

        if G2 == 0 or G2 is None:
            counter += 1
            continue
        else:
            # Get the polarity_dict:
            for comment in comments_controversial[i]:
                polarity_dict[comment['author']] = float(comment['sentiment']) if comment['sentiment'] != '' else 0.0
            for comment in comments_controversial[i + 1]:
                polarity_dict[comment['author']] = float(comment['sentiment']) if comment['sentiment'] != '' else 0.0

            save_to_file2(i+1, 1)

            polarization_score = calculate_polarization_score(G2, polarity_dict, num_walks, walk_length)
            print("Controversial: Polarization score - Random walk - for posts {} and {}: {}".format(i+1, i + 2, polarization_score))

            # Detect groups based on agreement using SCG
            # groups = SCG(G2, kmeans)

            polarization_score2 = compute_polarization_score_from_edges(G2)
            print("Controversial: Polarization score - From edges - for posts {} and {}: {}".format(i + 1, i + 2, polarization_score2))
            print("---------------------------------------------")
            # Print the nodes in each group
            # for j, group in enumerate(groups):
            #     print(f"Group {j + 1}: {group.nodes()}")


num_walks = 100  # Number of random walks to perform from each node
walk_length = 10  # Length of each random walk
k = 3  # The number of the groups I want to detect

# Run the processes
results_for_one_graph(num_walks, walk_length, k)
print("#############################################")
print("#############################################")
results_for_two_graphs(num_walks, walk_length, k)


