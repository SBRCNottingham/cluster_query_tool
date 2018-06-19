import networkx as nx
import joblib
import random
from . import louvain
from .louvain_consensus import create_partition_from_edge_set, modularity


def partitions(graph, seed):
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
    local_optima = louvain.best_partition(graph, partition=start_partition)

    q_s = modularity(graph, start_partition)
    q_o = modularity(graph, local_optima)

    return (q_s, start_partition), (q_o, local_optima)


def gen_sample(network_path, nsamples=10, seed=1):
    """
    Sampler for visualisation of network parition space
    :param network_path:
    :param nsamples:
    :param seed:
    :return:
    """
    graph = nx.read_edgelist(network_path)

    # Sample start and end partitions and modularity for them
    # store only unique partitions

    partition_results = joblib.Parallel(n_jobs=joblib.cpu_count())(joblib.delayed(partitions)(graph, s)
                                                                   for s in range(seed, seed+nsamples))

    with open("partitions_mod.txt") as of:
        # Output
        for start, lo in partition_results:
            for q, part in [start, lo]:
                ptl = str([int(part[node]) for node in graph.nodes()])
                of.write("{},{}\n".format(q, ptl))
