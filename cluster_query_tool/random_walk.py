import networkx as nx
import numpy as np


def diffusion_kernel(graph, query_nodes, beta=0.1):
    assert isinstance(graph, nx.Graph)

    l = np.array(nx.laplacian_matrix(graph).todense())
    diff_k = np.exp(-beta * l)

    score = np.zeros(graph.number_of_nodes())
    for i in query_nodes:
        score += diff_k[i]
    return score


def rwr(graph, query_nodes, restart_prob, _dthreshold=10e-7):
    # Get adjacency matrix
    a = nx.to_numpy_array(graph)

    # Normalise the column vectors
    nr = a.sum(axis=1)
    w = a / nr

    # initialise p_o
    p_o = np.zeros(graph.number_of_nodes())
    # p_o is the set of "seed" nodes that the walker would randomly jump to
    np.put(p_o, query_nodes, 1.0 / len(query_nodes))
    # Delta (change in L1 norm between p_t an p_tp1) should approach zero when the walker is at steady state
    # in other words Delta = 0 after an infinite random walk
    delta = 1.0
    # Initialise until p_t
    p_t = np.copy(p_o)
    # restart is a fixed vector (i.e. the probability of jumping back to one of the seed nodes)
    restart = p_o * restart_prob
    while delta > _dthreshold:
        p_tp1 = ((1 - restart_prob) * np.dot(w, p_t)) + restart
        delta = np.linalg.norm(p_t - p_tp1, 1)
        p_t = p_tp1

    return p_t
