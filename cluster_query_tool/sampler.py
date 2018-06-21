import networkx as nx
from joblib import Parallel, cpu_count, delayed
import random
from cluster_query_tool import louvain
from cluster_query_tool.louvain_consensus import create_partition_from_edge_set, partition_to_cut_set
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
        q_s = louvain.modularity(pl, graph)
        partitions.append((q_s, [float(pl[node] + 1) for node in graph.nodes()]))

    return partitions


def gen_sample(graph, nsamples=10000, seed=1, opt="partitions_mod.txt"):
    """
    Sampler for visualisation of network parition space
    :param network_path:
    :param nsamples:
    :param seed:
    :return:
    """

    # Sample start and end partitions and modularity for them
    # store only unique partitions

    partition_results = Parallel(n_jobs=cpu_count())(delayed(getpartitions)(graph, s)
                                                     for s in range(seed, seed +nsamples))

    cutsets = []
    with open(opt, "w+") as of:
        # Output
        for q, ptl in chain(*partition_results):
            cs = tuple(ptl)
            if cs not in cutsets:
                cutsets.append(cs)
                of.write("{},{}\n".format(q, ptl))


def sample_file(network_path):
    graph = nx.read_edgelist(network_path, nodetype=int)
    gen_sample(graph)
