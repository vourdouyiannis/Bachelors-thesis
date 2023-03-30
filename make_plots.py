# VOURDOUGIANNIS DIMITRIOS 4326 #

import json
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.shortest_paths.generic import shortest_path

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


def create_graph(ctr, post):
    # Create an empty graph
    G = nx.Graph()

    # Add nodes to the graph
    for comment in post:
        G.add_node(comment['author'])

    # Add edges to the graph
    create_edges(G, post)

    # Remove the nodes we don't need or causing a problem
    remove_nodes(G, post)

    # Draw the graph
    pos = nx.spring_layout(G, k=0.1)
    nx.draw(G, pos, node_size=1, node_color='black', edge_color='gray')

    # Save each graph to a file
    filename = "graphs/Post{}_graph.png".format(ctr)
    plt.savefig(filename)
    plt.clf()


counter = 1
for post in comments:
    create_graph(counter, post)
    counter += 1
