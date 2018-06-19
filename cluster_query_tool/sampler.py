import networkx as nx
import joblib
import random
from . import louvain
from .louvain_consensus import create_partition_from_edge_set, modularity
from itertools import chain


def getpartitions(graph, seed):
    """
    Generate a partition with louvain. If cut_set is None the starting partition will be randomly chosen.
    :param graph: nx.Graph
    :param cut_set: iterble, tuple of edges
    :param seed: random seed
    :return:
    """
    random.seed(seed)
    cut_set = random.sample(graph.edges(), random.randint(1, graph.number_of_edges()))
    # convert to partition
    start_partition = create_partition_from_edge_set(graph, cut_set)
    dend = louvain.generate_dendogram(graph, start_partition)

    partitions = []
    for i in range(len(dend)):
        pl = louvain.partition_at_level(dend, i)
        q_s = modularity(graph, pl)
        partitions.append((q_s, [int(pl[node]) for node in graph.nodes()]))

    return partitions


def gen_sample(network_path, nsamples=10, seed=1, opt="partitions_mod.txt"):
    """
    Sampler for visualisation of network parition space
    :param network_path:
    :param nsamples:
    :param seed:
    :return:
    """
    graph = nx.read_edgelist(network_path, nodetype=int)

    # Sample start and end partitions and modularity for them
    # store only unique partitions

    partition_results = getpartitions(graph, 1)
    with open(opt) as of:
        # Output
        for q, ptl in chain(*partition_results):
            of.write("{},{}\n".format(q, ptl))
