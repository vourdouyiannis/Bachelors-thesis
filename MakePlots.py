# VOURDOUGIANNIS DIMITRIOS 4326 #

import json
import networkx as nx
import matplotlib.pyplot as plt

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


def create_graph(post):
    # Create an empty graph
    G = nx.Graph()

    # Add nodes to the graph
    for comment in post:
        G.add_node(comment['author'])

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
    merged_graph = nx.disjoint_union(graph1, graph2)

    # Set nodes color
    node_color = []
    for node in nodes1:
        if node in common_nodes:
            node_color.append('purple')
        else:
            node_color.append('blue')
    for node in nodes2:
        if node in common_nodes:
            node_color.append('purple')
        else:
            node_color.append('red')

    # Set nodes position
    pos = nx.spring_layout(merged_graph, k=0.01)

    # for node in merged_graph.nodes():
    #     if node in graph1.nodes():
    #         pos[node] = (pos[node][0] - 0.5, pos[node][1])
    #         print("here")

    # # Connect the nodes that have commented in both posts
    # for node1 in nodes1:
    #     for node2 in nodes2:
    #         if node1 == node2:
    #             merged_graph.add_edge(node1, node2, color='green')
    #             print(node1, node2)

    # for node1 in merged_graph:
    #     for node2 in merged_graph:
    #         if node_color[list(merged_graph.nodes()).index(node1)] == 'purple' and node_color[list(merged_graph.nodes()).index(node2)] == 'purple':
    #             merged_graph.add_edge(node1, node2, color='green')

    # Draw the merged graph with different colors
    nx.draw(merged_graph, pos=pos, node_size=10, node_color=node_color, edge_color='gray')


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


counter = 1
for post in comments:
    G = create_graph(post)

    # Draw the graph
    pos = nx.spring_layout(G, k=0.05, scale=0.1, iterations=50)
    nx.draw(G, pos, node_size=2, node_color='blue', edge_color='gray')

    save_to_file(counter)
    counter += 1

counter = 1
# 0:no, 1:yes
controversial = 0
# 0:different, 1:same
subreddit = 1
for i in range(len(comments)-1):
    create_graph2(comments[i], comments[i+1])
    save_to_file2(counter, subreddit, controversial)
    counter += 1
