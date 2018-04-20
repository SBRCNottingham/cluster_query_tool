from __future__ import division
import numpy as np
cimport numpy as np


def fast_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
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


def mu_ivector(nodes, partitions, query):
    """
    For all nodes in V, return a vector $\mu$ such that $\mu$ is the fraction of times that a node is in
    the largest cluster that intersects with the query set over all clusters.

    returns muscore (numpy array of floats) and  mappings for keys to vector index.
    """
    muscore = np.zeros(nodes)

    for partition in partitions:
        cdef int best = 0
        best_c = []
        for cluster in partition:
            intersect = np.intersect1d(cluster, partition)
            if len(intersect) > best:
                best = len(intersect)
                best_c = cluster
            elif len(intersect) == best:
                best_c += cluster

        for i in best_c:
            muscore[i] += 1

    # Normalise result
    return muscore * 1/len(partitions)


def roc_score_seed(seed_set, nodes, index, comm):
    vec, key = mu_ivector(nodes, index, seed_set)
    inc = lambda x: 1 if x in comm else 0
    y_true = [inc(x) for x in nodes if x not in seed_set]
    y_score = [vec[key[x]] for x in nodes if x not in seed_set]
    return fast_auc(y_true, y_score)


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


def get_auc_scores_community(seed_size, community, graph, index, sample_seed=1337):
    np.random.seed(sample_seed)
    samples = unique_sampler(community, seed_size)
    auc_scores = Parallel(n_jobs=cpu_count())(delayed(roc_score_seed)(sample, graph, index, community)
                                              for sample in samples)
    return auc_scores