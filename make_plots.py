# VOURDOUGIANNIS DIMITRIOS 4326 #

import json
import networkx as nx
import matplotlib.pyplot as plt

# Read data from JSON file
with open('resources/comments.json', 'r') as f:
    comments = json.load(f)

def create_graph():
    counter = 1
    # Add nodes to the graph
    for post in comments:
        # Create an empty graph
        G = nx.Graph()

        # Add nodes to the graph
        for comment in post:
            G.add_node(comment['author'])

        # Add edges to the graph
        for i in range(len(post)):
            for j in range(len(post)):
                if post[i]['parent_id'] == post[j]['id']:
                    G.add_edge(post[i]['author'], post[j]['author'])

        # Draw the graph
        pos = nx.spring_layout(G, k=0.05)
        nx.draw(G, pos, node_size=1, node_color='black', edge_color='gray')

        # Save each graph to a file
        filename = "graphs/Post{}_graph.png".format(counter)
        plt.savefig(filename)
        plt.clf()
        counter += 1


create_graph()
