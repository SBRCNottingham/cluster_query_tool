from cluster_query_tool.louvain_consensus import mu_ivector

from sklearn.metrics import roc_auc_score

from cigram import lfr_benchmark_graph
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import random
import numpy as np
from cluster_query_tool.indexer import get_index
import logging

from scipy.special import binom
import itertools

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
    inc = lambda x: 1 if x in node_comms[n] else 0
    y_true = [inc(x) for x in graph.nodes()]
    y_score = [vec[key[x]] for x in graph.nodes()]

    return roc_auc_score(y_true, y_score)


def roc_score_seed(seed_set, graph, index, comm):
    vec, key = mu_ivector(graph, index, seed_set)
    inc = lambda x: 1 if x in comm else 0
    y_true = [inc(x) for x in graph.nodes()]
    y_score = [vec[key[x]] for x in graph.nodes()]
    return roc_auc_score(y_true, y_score)


def get_auc_scores_community(seed_size, community, graph, index, sample_seed=1337):
    np.random.seed(sample_seed)
    samples = unique_sampler(community, seed_size)
    auc_scores = Parallel(n_jobs=cpu_count())(delayed(roc_score_seed)(sample, graph, index, community)
                                              for sample in samples)
    return auc_scores

