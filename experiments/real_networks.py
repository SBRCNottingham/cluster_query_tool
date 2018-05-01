import networkx as nx
from cluster_query_tool import louvain_consensus
from cluster_query_tool.indexer import get_index
from experiments import fast_auc, unique_sampler
from joblib import Parallel, delayed, cpu_count
import json
from sklearn.metrics import roc_curve
import numpy as np
import os

real_networks = {
    "arabidopsis_ppi": {
        "path": "data/arabidopsis_ppi/arabadopsis_ppi.edgelist",
        "clusters": "data/arabidopsis_ppi/int_act_complexes.json",
        "index": "data/arabidopsis_ppi/index.json.xz",
        "node_type": "str",
    },
    "yeast_ppi": {
        "path": "data/yeast_ppi/yeast.edgelist",
        "clusters": "data/yeast_ppi/yeast_ppi.json",
        "index": "data/yeast_ppi/index.json.xz",
        "node_type": "str",
    },
    "eu_email": {
        "path": "data/eu_email/network.txt",
        "clusters": "data/eu_email/communities.json",
        "index": "data/eu_email/index.json.xz",
        "node_type": "int",
    }
}


def load_network(graph_path, graph_name, communities_path, index_path, node_type):
    """
    Load a real network into an nx Graph.
    Return tuple graph, communities, membership_matrix, node_mapping
    :param graph_path:
    :param graph_name:
    :param communities_path:
    :param index_path
    :return:
    """
    nt = str
    if node_type == "int":
        nt = int

    graph = nx.read_edgelist(graph_path, nodetype=nt)
    graph.name = graph_name

    with open(communities_path) as cp:
        comms = json.load(cp)

    index = get_index(graph, cache_location=index_path)
    mmatrix, nmap = louvain_consensus.membership_matrix(list(graph.nodes()), index)
    return graph, comms, mmatrix, nmap


def comm_significance(cid, comm, membership_matrix):
    return cid, louvain_consensus.quality_score(comm, membership_matrix)


def map_com(cnodes, nmap):
    return np.array([nmap[n] for n in cnodes])


def get_community_significance_scores(mmatrix, nmap, comms):
    """
    Return the community signifcance scores for a network
    :param mmatrix:
    :param nmap:
    :param comms:
    :return:
    """
    jobs = (delayed(comm_significance)(cid, map_com(comm, nmap), mmatrix) for cid, comm in comms.items())
    sign = Parallel(n_jobs=cpu_count())(jobs)
    return dict(sign)


def samp_roc_and_auc(samp, mmatrix, comm, cid, s):
    ncom = np.array([x for x in range(mmatrix.shape[0]) if x not in samp])
    vec = louvain_consensus.query_vector(np.array(samp), mmatrix)

    def inc(x):
        if x in comm:
            return 1
        return 0

    y_true = np.array([inc(x) for x in ncom])
    y_prob = vec[ncom]
    tpr, fnr, _ = roc_curve(y_true, y_prob)
    auc = fast_auc(y_true, y_prob)
    return cid, s, tpr, fnr, auc


def gen_roc_curves(mmatrix, comm, s, cid):
    funcs = (delayed(samp_roc_and_auc)(samp, mmatrix, comm, cid, s) for samp in unique_sampler(comm, s))
    results = Parallel(n_jobs=cpu_count(), backend='threading')(funcs)
    return list(results)


def get_rocs(mmatrix, nmap, comms, seed_sizes=(1, 3, 7, 15)):
    """

    :param mmatrix:
    :param nmap:
    :param comms:
    :param seed_sizes:
    :return:
    """

    res = []
    for cid, comm in comms.items():
        cnodes = map_com(comm, nmap)
        for s in seed_sizes:
            if len(comm) > s:
                res += gen_roc_curves(mmatrix, cnodes, s, cid)
    return res


def plot_roc_curve(roc_df, save_path):
    """
    Takes a dataframe of roc curves at different seeds and plots the mean roc curves

    Separates the results into different seed set sizes
    computes an interpolated curve along with standard deviations in coloured shaded areas.

    Returns matplotlib axis and fig objects

    Labels incluse the mean AUC with standard deviations
    :param roc_df:
    :param save_path:
    :return:
    """
    pass


def generate_results(network, overwrite=False):
    dt = real_networks[network]
    graph, comms, mmatrix, nmap = load_network(dt["path"], network, dt["clusters"], dt["index"], dt["node_type"])

    roc_df_path = os.path.join("results", graph.name) + "roc_res.json"

    if overwrite or not os.path.exists(roc_df_path):
        print(network, "gen_roc_curves")
        roc_results = get_rocs(mmatrix, nmap, comms)
        with open(roc_df_path, "w+") as roc_df:
            json.dump(roc_results, roc_df)

    sign_df_path = os.path.join("results", graph.name) + "sign_res.json"

    if overwrite or not os.path.exists(sign_df_path):
        sigscores = get_community_significance_scores(mmatrix, nmap, comms)
        print(network, "gen_sig_scores")
        with open(sign_df_path, "w+") as sig_df:
            json.dump(sigscores, sig_df)


if __name__ == "__main__":
    for n in real_networks:
        print(n)
        generate_results(n)
