from cluster_query_tool.louvain_consensus import query_vector

from cigram import lfr_benchmark_graph
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import random
import numpy as np
from cluster_query_tool.indexer import get_index
import logging
from cluster_query_tool.random_walk import rwr

from scipy.special import binom
import itertools

from numba import jit

logger = logging.getLogger("cigram.generators")
logger.setLevel(logging.ERROR)


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


def get_benchmark(params, num_procs=4):
    graph, communities = lfr_benchmark_graph(**params)
    return graph, communities, get_index(graph, num_procs=num_procs)


def unique_sampler(node_set, sample_size, max_samples=120):
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


@jit(nopython=True, nogil=True)
def fast_auc(y_true, y_prob):
    """
    Numba implementation of auc score calculation for multi threaded purposes
    :param y_true:
    :param y_prob:
    :return:
    """
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc


def roc_score_seed(seed_set, nodes, membership_ma, comm, tmm):
    vec = query_vector(seed_set, membership_ma)

    y_true = get_y_true(seed_set, tmm)
    y_score = np.array([vec[x] for x in nodes if x not in seed_set])
    return fast_auc(y_true, y_score)


def roc_score_seed_rwr(seed_set, graph, comm, nmap, tmm):
    ncom = np.array([nmap[x] for x in graph.nodes() if nmap[x] not in seed_set])
    probs = rwr(graph, seed_set)

    y_true = get_y_true(seed_set, tmm)
    y_score = probs[ncom]
    return fast_auc(y_true, y_score)


def get_auc_scores_community(seed_size, community, nodes, membership_ma, tmm, sample_seed=1337):
    np.random.seed(sample_seed)
    samples = unique_sampler(community, seed_size)

    funcs = (delayed(roc_score_seed)(np.array(sample), nodes, membership_ma, community, tmm) for sample in samples)
    auc_scores = Parallel(n_jobs=cpu_count(), backend='threading')(funcs)
    return auc_scores


def get_auc_scores_community_rwr(seed_size, community, graph, nmap, tmm, sample_seed=1337):
    np.random.seed(sample_seed)
    samples = unique_sampler(community, seed_size)

    funcs = (delayed(roc_score_seed_rwr)(np.array(sample), graph, community, nmap, tmm) for sample in samples)
    auc_scores = Parallel(n_jobs=cpu_count(), backend='threading')(funcs)
    return auc_scores


def construct_true_memberships_matrix(nmap, com_dict):
    """
    Given a dict of communities and node mapping, return a N x |C| binary matrix of memberships
    Also returns a dict of keys for the community indexes
    :param nmap:
    :param com_dict:
    :return:
    """
    tmm = np.zeros((len(nmap), len(com_dict)), dtype=np.int8)

    c_keys = dict(enumerate(sorted(com_dict.keys())))

    for i, cid in c_keys.items():
        for n in com_dict[cid]:
            tmm[nmap[n]][i] = 1

    return tmm, c_keys


def get_y_true(seed_ind, tmm):
    """
    returns true membership vector for a set of seed nodes
    This corrects for the situation where a group of seed nodes have the same overlapping community membership
    """
    if len(seed_ind) == 1:
        seed_ind = np.array([seed_ind[0], seed_ind[0]])

    subm = tmm[np.array(seed_ind)]

    # Recursive bitwise and
    tv = np.bitwise_and(subm[0], subm[0])
    for sb in subm[1:]:
        tv = np.bitwise_and(tv, sb)

    comm_indexes = np.where(tv)

    tr = np.unique(np.where(tmm.transpose()[comm_indexes]))

    y_true = np.array(
        [i in tr for i in range(tmm.shape[0]) if i not in seed_ind],
        dtype=np.int8
    )
    return y_true