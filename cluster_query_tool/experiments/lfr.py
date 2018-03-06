from cigram import lfr_benchmark_graph
from cluster_query_tool.louvain_consensus import mu_ivector
from cluster_query_tool.indexer import get_index
from cluster_query_tool.experiments import unique_sampler
from sklearn.metrics import roc_auc_score, average_precision_score


def gen_network_index(params):
    """

    :param params:
    :return: partitions - ensemble of network partitions
    """
    graph, communities = lfr_benchmark_graph(**params)
    # Partitions are cached
    partitions = get_index(graph)
    return partitions


def auc_scores(gparams, partitions, seed_size=3, max_samples=120):
    """

    :param graph:
    :param comms:
    :param partitions:
    :param seed_size:
    :param max_samples:
    :return:
    """

    graph, comms = lfr_benchmark_graph(**gparams)
    auc_scores = dict()
    app_scores = dict()

    for c in comms:
        for cs in unique_sampler(comms[c], seed_size, max_samples=max_samples):
            vec, key = mu_ivector(graph, partitions, cs)
            inc = lambda x: 1 if x in comms[c] else 0
            y_true = [inc(x) for x in graph.nodes() if x not in cs]
            y_score = [vec[key[x]] for x in graph.nodes() if x not in cs]

            auc_scores[c].append(roc_auc_score(y_true, y_score))
            app_scores[c].append(average_precision_score(y_true, y_score))


def sampler(seed, num_jobs, **kwargs):
    """"""
    Parallel(num_jobs=num_jobs)