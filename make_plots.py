# VOURDOUGIANNIS DIMITRIOS 4326 #

import json
import networkx as nx
import matplotlib.pyplot as plt

# Read data from JSON file
with open('comments.json', 'r') as f:
    comments = json.load(f)

# Create an empty graph
G = nx.Graph()


# Add nodes to the graph
for post in comments:
    for comment in post:
        G.add_node(comment['author'], label=comment['author'])

# Add edges to the graph
# for edge in comments['edges']:
#     G.add_edge(edge['source'], edge['target'])

# Draw the graph
pos = nx.spring_layout(G, k=0.05)
nx.draw(G, pos, node_size=1, node_color='black', edge_color='gray', with_labels=False)

# Show the graph
plt.show()
