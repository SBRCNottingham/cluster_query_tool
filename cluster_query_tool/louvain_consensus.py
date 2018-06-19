from __future__ import division
from . import louvain
import networkx as nx
import random
import numpy as np
from numba import jit


def modularity(graph, part):
    """
    Return the newman modularity of graph graph
    Optional parameter part, is a partition of a graph
    """
    partition = dict()
    for k, com in enumerate(part):
        for node in com:
            partition[node] = k

    return louvain.modularity(partition, graph)


def create_partition_from_edge_set(graph, edge_set, parition_nodes_format=False):
    """
    Cut out connected components
    """
    graph.remove_edges_from(edge_set)
    partition = {}
    partition_map = {}
    for i, cc in enumerate(nx.connected_components(graph)):
        partition[i] = cc
        for node in cc:
            partition_map[node] = i

    graph.add_edges_from(edge_set)
    if parition_nodes_format:
        return partition

    return partition_map


def partition_to_cut_set(graph, partition):
    """
    Partition in {node:community} format
    Fast
    # Returns a sorted tuple of edge tuples, hashable
    """
    return tuple(sorted(set([edge for edge in graph.edges() if partition[edge[0]] != partition[edge[1]]])))


def gen_local_optima_community(graph, cut_set=None):
    """
    Generate a partition with louvain. If cut_set is None the starting partition will be randomly chosen.
    :param graph: nx.Graph
    :param cut_set: iterble, tuple of edges
    :return:
    """
    # The possible space of communities is huge!
    if cut_set is None:
        cut_set = random.sample(graph.edges(), random.randint(1, graph.number_of_edges()))
    # convert to partition
    start_partition = create_partition_from_edge_set(graph, cut_set)
    local_optima = louvain.best_partition(graph, partition=start_partition)

    return start_partition, local_optima


def membership_matrix(nodes, partitions):
    nmap = dict(((n, m) for m, n in enumerate(sorted(nodes))))
    M = np.zeros((len(nodes), len(partitions)), dtype=np.int16)

    for pi, partition in enumerate(partitions):
        for ci, cluster in enumerate(partition, 1):
            for v in cluster:
                M[nmap[v]][pi] = ci
    return M, nmap


@jit(nopython=True, nogil=True)
def query_vector(query_indexes, M):
    qm = np.zeros(M.shape[0])
    norm_const = 1 / (M.shape[1] * query_indexes.shape[0])
    for v in np.ndindex(M.shape[0]):
        for q in query_indexes:
            # Number of times node is in the same cluster as a query node
            qm[v] += (M[v] == M[q]).sum()

    return qm * norm_const
