import networkx as nx
from cluster_query_tool import louvain_consensus
from cluster_query_tool.indexer import get_index
from experiments import fast_auc, unique_sampler
from joblib import Parallel, delayed, cpu_count
import json
import pickle
from sklearn.metrics import roc_curve
import numpy as np
import os
import progressbar
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy import interp


real_networks = {
    "arabidopsis_ppi": {
        "path": "data/arabidopsis_ppi/arabidopsis_ppi.edgelist",
        "clusters": "data/arabidopsis_ppi/int_act_complexes.json",
        "index": "data/arabidopsis_ppi/index.json.xz",
        "node_type": "str",
    },

    "ecoli_ppi": {
        "path": "data/ecoli_ppi/ecoli_ppi.edgelist",
        "clusters": "data/ecoli_ppi/ecoli_ppi.json",
        "index": "data/ecoli_ppi/index.json.xz",
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

    rcomms = dict()
    for c in comms:
        nc = [x for x in comms[c] if x in nmap]
        if len(nc) > 2:
            rcomms[c] = nc

    return graph, rcomms, mmatrix, nmap


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
    sign = Parallel(n_jobs=cpu_count())(delayed(comm_significance)(cid, map_com(comm, nmap), mmatrix)
                                        for cid, comm in comms.items())
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


def samp_auc(samp, mmatrix, comm, cid, s, mat_size):
    ncom = np.array([x for x in range(mmatrix.shape[0]) if x not in samp])
    vec = louvain_consensus.query_vector(np.array(samp), mmatrix)

    def inc(x):
        if x in comm:
            return 1
        return 0

    y_true = np.array([inc(x) for x in ncom])
    y_prob = vec[ncom]
    auc = fast_auc(y_true, y_prob)
    return cid, s, mat_size, auc


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

    for cid, comm in progressbar.progressbar(comms.items()):
        cnodes = map_com(comm, nmap)
        for s in seed_sizes:
            if len(comm) > s:
                res += gen_roc_curves(mmatrix, cnodes, s, cid)
    return res


def psize_scoring(mmatrix, nmap, comms, seed_sizes=(1, 3, 7, 15)):
    funcs = []
    for mat_size in np.linspace(10, 2000, 10):
        sub_mat = mmatrix.transpose()[:int(mat_size)].transpose()
        for cid, comm in comms.items():
            cnodes = map_com(comm, nmap)
            for s in seed_sizes:
                samples = unique_sampler(cnodes, s)
                if len(comm) > s:
                    for samp in samples:
                        funcs.append(delayed(samp_auc)(samp, sub_mat, cnodes, cid, s, mat_size))

    results = Parallel(n_jobs=cpu_count(), verbose=5)(funcs)

    return results


def plot_roc_curve(df, save_path):
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
    colours = {
        1: "b",
        3: "g",
        7: "y",
        15: "m"
    }

    fig, ax = plt.subplots()
    fig.set_dpi(90)

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.01])
    ax.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), '--', color="#cccccc")

    for s in df["seed"].unique():

        sdf = df.loc[df["seed"] == s]

        tprs = []
        base_fpr = np.linspace(0, 1, 101)

        for rid, row in sdf.iterrows():
            tpr = interp(base_fpr, row["tpr"], row["fnr"])
            tpr[0] = 0.0
            tprs.append(tpr)

        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        std = tprs.std(axis=0)

        tprs_upper = np.minimum(mean_tprs + std, 1)
        tprs_lower = mean_tprs - std

        label = "{} seed node(s) mean AUC: {:.2f} (std: {:.2f})".format(s, sdf.mean()["auc"], sdf.std()["auc"])

        ax.plot(base_fpr, mean_tprs, color=colours[s], label=label)
        ax.fill_between(base_fpr, tprs_lower, tprs_upper, color=colours[s], alpha=0.1)

        ax.legend()

    fig.tight_layout()
    fig.savefig(save_path)


def generate_results(network, overwrite=False):
    dt = real_networks[network]

    roc_df_path = os.path.join("results", network) + "_roc_res.p"

    if overwrite or not os.path.exists(roc_df_path):
        graph, comms, mmatrix, nmap = load_network(dt["path"], network, dt["clusters"], dt["index"], dt["node_type"])
        print(network, "gen_roc_curves")
        roc_results = get_rocs(mmatrix, nmap, comms)
        with open(roc_df_path, "wb+") as roc_df:
            pickle.dump(roc_results, roc_df)

    sign_df_path = os.path.join("results", network) + "_sign_res.p"

    if overwrite or not os.path.exists(sign_df_path):
        graph, comms, mmatrix, nmap = load_network(dt["path"], network, dt["clusters"], dt["index"], dt["node_type"])
        sigscores = get_community_significance_scores(mmatrix, nmap, comms)
        print(network, "gen_sig_scores")
        with open(sign_df_path, "wb+") as sig_df:
           pickle.dump(sigscores, sig_df)

    roc_res_partitions = os.path.join("results", network) + "auc_res_partitions.p"
    if overwrite or not os.path.exists(roc_res_partitions):
        graph, comms, mmatrix, nmap = load_network(dt["path"], network, dt["clusters"], dt["index"], dt["node_type"])
        psize_scores = psize_scoring(mmatrix, nmap, comms)

        with open(roc_res_partitions, "wb+") as auc_res_df:
           pickle.dump(psize_scores, auc_res_df)


def handle_results(network):

    dt = real_networks[network]

    roc_df_path = os.path.join("results", network) + "_roc_res.p"
    with open(roc_df_path, "rb") as rf:
        results = pickle.load(rf)

    headings = ["cid", "seed", "tpr", "fnr", "auc"]

    df = pd.DataFrame(results, columns=headings)
    plot_roc_curve(df, "article/images/rocs/{}.png".format(network))
    plot_roc_curve(df, "article/images/rocs/{}.svg".format(network))
    plot_roc_curve(df, "article/images/rocs/{}.eps".format(network))
    return df


if __name__ == "__main__":
    for n in real_networks:
        print(n)
        generate_results(n)
        handle_results(n)
