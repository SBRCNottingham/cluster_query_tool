"""
Create and save a partition space index for the data
"""
from cluster_query_tool.louvain_consensus import gen_local_optima_community, partition_to_cut_set
import random
from multiprocessing import cpu_count
import json
import os
from joblib import Parallel, delayed
import networkx as nx


def _try_int(num):
    """
    Can the value be parsed as an int?
    :param num: any type
    :return: Boolean
    """
    try:
        int(num)
    except ValueError:
        return False

    return True


def get_index(graph, space_sample_size="medium", seed=1, num_procs=cpu_count(), use_cache=True):
    """
    Generate a partition space index for a given network
    :param graph: networkx graph object (make sure name is set for caching)
    :param space_sample_size: can be string high (10,000 samples), medium (2000 samples) or low (100 samples) or int of
    samples
    :param seed:
    :param num_procs:
    :param use_cache: cache result to disk
    :return:
    """
    # TODO: change the actual values of these numbers to something more solid
    if space_sample_size == "low":
        space_sample_size = 100
    elif space_sample_size == "medium":
        space_sample_size = 2000
    elif space_sample_size == "high":
        space_sample_size = 10000
    elif not _try_int(space_sample_size):
        space_sample_size = 2000

    if not os.path.exists(".ctq_cache"):
        os.mkdir(".ctq_cache")

    space_sample_size = int(space_sample_size)
    # cache_result
    cache_location = os.path.join(".ctq_cache", "get_index_{0}_{1}_{2}.json".format(seed,
                                                                                    space_sample_size, graph.name))

    if use_cache and os.path.exists(cache_location):

        try:
            with open(cache_location) as cache:
                result = json.load(cache)
            return result
        except json.JSONDecodeError:
            pass

    result = gen_sample_sets(graph, num_samples=space_sample_size, num_procs=num_procs, seed=seed)

    if use_cache:
        with open(cache_location, "w+") as cache:
            json.dump(result, cache)
        
    return result


def partition_from_cut(graph, edge_set):
    """
    Cut out connected components
    """
    graph.remove_edges_from(edge_set)
    partition = []
    for i, cc in enumerate(nx.connected_components(graph)):
        partition.append(tuple(cc))

    return partition


def _sample_local_optima(graph, seed):
    """

    :param graph:
    :param seed:
    :return:
    """
    random.seed(seed)
    start_partition, local_optima = gen_local_optima_community(graph)
    # Hashable type
    cut_set = partition_to_cut_set(graph, local_optima)
    # Sometimes we generate an initial cut set that reduces to empty set

    return cut_set


def gen_sample_sets(graph, num_procs=cpu_count(), num_samples=2000, seed=1):
    """
    Compute louvain accross multiple cores
    """
    cut_sets = Parallel(n_jobs=num_procs)(delayed(_sample_local_optima)(graph, s)
                                            for s in range(seed, seed+num_samples))

    cut_sets = list(set(cut_sets))
    # Only partition unique cut sets
    partitions = Parallel(n_jobs=num_procs)(delayed(partition_from_cut)(graph, cs) for cs in cut_sets)

    return [list(p) for p in partitions]
