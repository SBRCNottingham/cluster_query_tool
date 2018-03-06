from scipy.special import binom
import itertools


def unique_sampler(node_set, sample_size, max_samples=120):
    """
    For sampling random query sets of a given size.

    The node set size (e.g. true positives) limits the maximum number of samples to n choose k
    where n is the node set size and k is the size of the sample

    we want up to max_samples unique sets to be used as random starting labels

    :param node_set: Set of nodes to sample from
    :param sample_size: How many nodes to sample
    :param max_samples: Maximum number of samples
    :return:
    """
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
