from cluster_query_tool.louvain_consensus import mu_ivector

from sklearn.metrics import roc_auc_score

from cigram import lfr_benchmark_graph
from joblib import Parallel, delayed, dump, load
from multiprocessing import cpu_count
import random
import numpy as np
from cluster_query_tool.indexer import get_index
import logging

from scipy.special import binom
import itertools
from numba import jit

logger = logging.getLogger("cigram.generators")
logger.setLevel(logging.WARNING)


def param_modifier(params, param, values):
    for val in values:
        nparams = params.copy()
        nparams[param] = val
        yield nparams


def _set_seed(params):
    params["seed"] = random.randint(1, 10000000)
    return params


def gen_benchmark(params, seed, samples, n_jobs=cpu_count()):
    random.seed(seed)
    res = Parallel(n_jobs=n_jobs)(delayed(lfr_benchmark_graph)(**_set_seed(params)) for _ in range(samples))
    return res


def get_benchmarks(base_params, modified_params, param_vals, seed, samples):
    for pset in param_modifier(base_params, modified_params, param_vals):
        benchmarks = gen_benchmark(pset, seed, samples)
        for graph, communities in benchmarks:
            yield graph, communities, get_index(graph)


def get_benchmark(params):
    graph, communities = lfr_benchmark_graph(**params)
    return graph, communities, get_index(graph)


def unique_sampler(node_set, sample_size, max_samples=96):
    sample_sets = set()

    # maximum number of samples
    sspace = binom(len(node_set), sample_size)
    if sspace > max_samples:

        if sspace > max_samples * 2:
            # problem with this loop is when max_samples is close to n choose k
            while len(sample_sets) < max_samples:
                samp = tuple(sorted(np.random.choice(node_set, sample_size, replace=False)))
                sample_sets.add(samp)
        else:
            # Generate max_samples, unique random numbers sampling without replacement from possible combinations
            # Faster than above loop in case where n choose k is close to max_samples
            sp = np.random.choice(range(int(binom(len(node_set), sample_size))), max_samples, replace=False)
            poss = list(itertools.combinations(node_set, sample_size))
            for s in sp:
                sample_sets.add(poss[s])

    else:
        sample_sets = set(itertools.combinations(node_set, sample_size))

    return sample_sets


def roc_score_node(n, graph, index, node_comms):
    vec, key = mu_ivector(graph, index, [n])

    def inc(x):
        if x in node_comms[n]:
            return 1
        return 0

    y_true = [inc(x) for x in graph.nodes() if x != n]
    y_score = [vec[key[x]] for x in graph.nodes() if x != n]

    return roc_auc_score(y_true, y_score)


@jit
def mu_ivectord(nodes, partitions, query_set):
    """
    For all nodes in V, return a vector $\mu$ such that $\mu$ is the fraction of times that a node is in
    the largest cluster that intersects with the query set over all clusters.

    returns muscore (numpy array of floats) and  mappings for keys to vector index.
    """
    muscore = np.zeros(len(nodes))

    for partition in partitions:
        best = max(partition * query_set, key=np.sum)
        muscore += 1 * best

    # Normalise result
    retval = np.array(muscore * 1/len(partitions))
    return retval


def roc_score_seed(seed_set, nodes, index, comm):
    vec = mu_ivectord(nodes, index, np.array([n in seed_set for n in nodes]))

    key = dict(((j, i) for i, j in enumerate(nodes)))
    y_true = [x in comm for x in nodes if x not in seed_set]
    y_score = [vec[key[x]] for x in nodes if x not in seed_set]
    return roc_auc_score(y_true, y_score)


def get_auc_scores_community(seed_size, community, nodes, index, sample_seed=1337):
    np.random.seed(sample_seed)
    samples = unique_sampler(community, seed_size)

    auc_scores = Parallel(n_jobs=cpu_count())(delayed(roc_score_seed)(sample, nodes, index, community)
                                              for sample in samples)

    return np.mean(auc_scores), np.std(auc_scores)


def index_to_one_hot_encoding(nodes, index):
    enc = []
    for partition in index:
        h_encoding = []
        for cluster in partition:
            h_encoding.append(np.array([n in cluster for n in nodes], dtype=np.bool))

        enc.append(np.array(h_encoding))

    return np.array(enc)
