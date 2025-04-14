import networkx as nx
import matplotlib.pyplot as plt
from graph_transformer import local_compl, transform_xyyx, transform_xzzx


# make linear chain graph
g = nx.Graph()
for i in range(13):
    g.add_edge(i, i+1)
nx.draw(g, with_labels=True)
nx.set_node_attributes(g, '', 'LC')  # store additional local Clifford (LC) gates as node attribute of graph (initialize to '' meaning identity)
plt.show()

# apply some local complementations to get the initial state from Fig. 3(a) in https://arxiv.org/pdf/2405.02414
local_compl(g, 3)
local_compl(g, 6)
local_compl(g, 7)
local_compl(g, 6)
local_compl(g, 10)
nx.draw(g, with_labels=True)
plt.show()

# apply fusions to get the cube graph
transform_xyyx(g, 3, 10)
transform_xzzx(g, 0, 6)
transform_xzzx(g, 7, 13)
nx.draw(g, with_labels=True)
plt.show()

