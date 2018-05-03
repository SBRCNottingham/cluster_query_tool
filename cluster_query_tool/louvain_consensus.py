from __future__ import division
from . import louvain
import networkx as nx
import random
import numpy as np
from scipy.stats import mannwhitneyu
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


def mu_ivector(graph_or_nodes, partitions, query_nodes):
    """
    For all nodes in V, return a vector $\mu$ such that $\mu$ is the fraction of times that a node is in
    the largest cluster that intersects with the query set over all clusters.

    returns muscore (numpy array of floats) and  mappings for keys to vector index.
    """
    nodes = {}
    if isinstance(graph_or_nodes, nx.Graph):
        nodes = set(graph_or_nodes.nodes())
    else:
        nodes = set(graph_or_nodes)

    query_set = set(query_nodes)
    muscore = np.zeros(len(nodes))

    key = dict([(k, i) for i, k in enumerate(sorted(nodes))])

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


def membership_matrix(nodes, partitions):
    nmap = dict(((n, m) for m, n in enumerate(sorted(nodes))))
    M = np.zeros((len(nodes), len(partitions)), dtype=np.int16)

    for pi, partition in enumerate(partitions):
        for ci, cluster in enumerate(partition, 1):
            for v in cluster:
                M[nmap[v]][pi] = ci
    return M, nmap


def quality_score(query_indexes, M):
    """

    :param query_indexes:
    :param M:
    :return:
    """
    vec = query_vector(query_indexes, M)
    nset = np.array([x for x in range(vec.shape[0]) if x not in query_indexes])
    u_score, pvalue = mannwhitneyu(vec[nset], vec[query_indexes])
    return pvalue


@jit(nopython=True, nogil=True)
def query_vector(query_indexes, M):
    qm = np.zeros(M.shape[0])
    norm_const = 1 / (M.shape[1] * query_indexes.shape[0])
    for v in np.ndindex(M.shape[0]):
        for q in query_indexes:
            # Number of times node is in the same cluster as a query node
            if q != v:
                qm[v] += (M[v] == M[q]).sum()

    return qm * norm_const


@jit(nopython=True, nogil=True)
def mui_vec_membership(query_indexes, membership_mat):
    # Get the sub matrix slice based on query node indexes
    # qmat = membership_mat[np.array(query_indexes, dtype=np.int16)]
    '''
    qmat = np.zeros((query_indexes.shape[0], membership_mat.shape[1]))
    for i in range(query_indexes.shape[0]):
        qmat[i] = membership_mat[query_indexes[i]]
    '''
    qmat = membership_mat[query_indexes]

    qtrans = qmat.transpose()
    mu_vec = np.zeros(membership_mat.shape[0])

    mtrans = membership_mat.transpose()

    for col_id in range(qtrans.shape[0]):
        for it in range(1, np.max(qtrans[col_id]) + 1):
            # size of intersection for each community
            isize = (qtrans[col_id] == it).sum()
            mu_vec[np.where((mtrans[col_id] == it))] += isize
    mu_vec *= 1 / (qmat.shape[0] * membership_mat.shape[1])
    return mu_vec


@jit(nopython=True, nogil=True)
def mui_vec_largest_intersec(query_indexes, membership_mat):
    # Get the sub matrix slice based on query node indexes
    qmat = membership_mat[query_indexes]

    qtrans = qmat.transpose()
    mu_vec = np.zeros(membership_mat.shape[0])

    mtrans = membership_mat.transpose()

    for col_id in np.ndindex(qtrans.shape[0]):

        maxi = 0
        max_it = [0]
        for it in range(1, np.max(qtrans[col_id]) + 1):
            # size of intersection for each community
            isize = (qtrans[col_id] == it).sum()
            if isize > maxi:
                maxi = isize
                max_it = [int(it)]
            # In case of a tie for largest intersection
            elif isize == maxi:
                max_it += [int(it)]

        for it in max_it:
            mu_vec[np.where((mtrans[col_id] == it))] += 1
    mu_vec *= 1 / membership_mat.shape[1]
    return mu_vec


@jit(nopython=True)
def _unique(arr):
    it = np.array([1])
    for x in arr:
        if x not in it:
            it.append([x])
    return it


@jit(nopython=True, nogil=True)
def uniform_membership_probability(mmatrix, q_size):
    n = mmatrix.shape[0]
    prob = 0.0

    for i in mmatrix.transpose():
        for c in _unique(i):
            ps = ((i == c).sum() / n)
            for x in range(2, q_size + 1):
                prob += ps ** x

    return prob * 1/mmatrix.shape[1]
