import networkx as nx
from cluster_query_tool import louvain_consensus
from cluster_query_tool.indexer import get_index
from .experiments import fast_auc


def load_network(graph_path, graph_name, communities_path):
    """
    Load a real network into an nx Graph.
    Return tuple graph, communities, membership_matrix, node_mapping
    :param graph_path:
    :param graph_name:
    :param communities_path:
    :return:
    """
    pass


def get_community_significance_scores(graph, membership_matrix, nmap, communities):
    """
    Return the community signifcance scores for a network
    :param graph:
    :param membership_matrix:
    :param nmap:
    :param communities:
    :return:
    """
    pass


def get_roc_scores(graph, membership_matrix, nmap, communities):
    pass


def get_auc_scores(graph, membership_matrix, nmap, communities):
    pass


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
