# cluster query tool

This is a tool for the use of modular sturcture to find related items
within complex networks based on a small number of seed nodes.

The contents of this repository includes code for the research paper:

**From clusters to queries: exploiting the uncertainty in the modularity
landscape of complex networks** James P Gilbert and Jamie Twycross
MLG at KDD 2018.

Written for Python version 3.5+

## Installation
To install, create a `virtualenv` environment and run

    $ python setup.py install

## Example Usage

Currently, the API is a little involved to work with.
However, here is a basic example of how to use the tool.

    from cluster_query_tool.indexer import get_index
    from cluster_query_tool.louvain_consensus import query_vector, membership_matrix
    import networkx as nx

    graph = nx.read_edgelist("data/eu_email/network.txt")
    # This may take some time but
    index = get_index(graph)

    memberships, nmap = memership_matrix(graph, index)

    # initial seed nodes
    query = [1, 2, 5 ,8]
    seed_set = np.array([nmap[x] for x in query])

    vec = query_vector(memberships, seed_set)

`vec` is an np array of values between 0 and 1 for the membership of node `i`
(`vec[nmap[i]]` is the value for index of node i).

