from __future__ import division
from . import louvain
import networkx as nx
import random
import numpy as np
from scipy.stats import mannwhitneyu
from itertools import combinations, product
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


def gen_local_optima_community(graph):
    # The possible space of communities is huge!
    sample = random.sample(graph.edges(), random.randint(1, graph.number_of_edges()))
    # convert to partition
    start_partition = create_partition_from_edge_set(graph, sample)
    local_optima = louvain.best_partition(graph, partition=start_partition)

    return start_partition, local_optima


def mu_iscore(nodes, partitions, query_nodes):
    """

    """
    query_set = set(query_nodes)
    qs = 0
    muscore = dict([(n, 0) for n in nodes])

    for partition in partitions:
        best = max([(len(query_set.intersection(set(cluster))), cluster) for cluster in partition], key=lambda x: x[0])

        qs += 1.0/(len(query_set) - 1) * (best[0] - 1)

        for n in best[1]:
            if n in muscore and best[0] > 0:
                muscore[n] += 1.0/len(partitions)

    qs *= 1.0/len(partitions)

    return muscore


def mu_ivector(graph, partitions, query_nodes):
    """
    For all nodes in V, return a vector $\mu$ such that $\mu$ is the fraction of times that a node is in
    the largest cluster that intersects with the query set over all clusters.

    returns muscore (numpy array of floats) and  mappings for keys to vector index.
    """
    query_set = set(query_nodes)
    muscore = np.zeros(graph.number_of_nodes())

    key = dict([(k, i) for i, k in enumerate(sorted(graph.nodes()))])

    for partition in partitions:
        best = max([(len(query_set.intersection(set(cluster))), cluster) for cluster in partition])

        for i in best[1]:
            muscore[key[i]] += 1

    # Normalise result
    return muscore * 1/len(partitions), key


def mu_ivector_n(graph, partitions, query_nodes):
    """
    For all nodes in V, return a vector $\mu$ such that $\mu$ is the fraction of times that a node is in
    the largest cluster that intersects with the query set over all clusters.

    returns muscore (numpy array of floats) and  mappings for keys to vector index.
    """
    query_set = set(query_nodes)
    muscore = np.zeros(graph.number_of_nodes())

    key = dict([(k, i) for i, k in enumerate(sorted(graph.nodes()))])

    for partition in partitions:
        for cluster in partition:
            s = len(query_set.intersection(cluster))
            if s > 0:
                for i in cluster:
                    muscore[key[i]] += 1/s

    # Normalise result
    return muscore * 1 / len(partitions), key


def q_significance(graph, index, testset):
    """
    For a given query set, return the p value from a mann-whitney u test that the null hypothesis that the query nodes
    could have been selected randomly from within the network.
    :param graph:
    :param index:
    :param testset:
    :return:
    """
    testset = set(testset)
    vec, keys = mu_ivector(graph, index, testset)

    network_dist = [vec[keys[x]] for x in keys if x not in testset]
    seed_dist = [vec[keys[x]] for x in keys if x in testset]
    u_score, pvalue = mannwhitneyu(network_dist, seed_dist)
    return pvalue


def s_quaility(query_nodes, nmap, M):
    Mx = []
    for i, j in combinations(set([nmap[i] for i in query_nodes]), 2):
        Mx.append((M[i] == M[j]).sum(axis=0) * 1 / M.shape[1])
    return np.mean(Mx), np.std(Mx)


def query_vector(query_nodes, nmap, M):
    qm = np.zeros(M.shape[0])
    norm_const = 1 / (M.shape[1] * len(query_nodes))
    for v in nmap.values():
        for q in query_nodes:
            qm[v] += (M[v] == M[nmap[q]]).sum()

    return qm * norm_const


@jit(nopython=True, nogil=True)
def query_vector_jit(query_indexes, M):
    qm = np.zeros(M.shape[0])
    norm_const = 1 / (M.shape[1] * query_indexes.shape[0])
    for v in range(M.shape[0]):
        for q in query_indexes:
            qm[v] += (M[v] == M[q]).sum()

    return qm * norm_const
