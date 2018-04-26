import click
import os
import json
import glob
from hashlib import sha256
from .indexer import partition_to_cut_set, partition_from_cut, dump
from .louvain_consensus import gen_local_optima_community
import random
import networkx as nx


def load_graph(path):
    graph = nx.read_edgelist(path)
    graph.name = os.path.basename(path)
    return graph


@click.command()
@click.argument('graph_path')
@click.argument('seed')
@click.argument('n_jobs')
@click.argument('results_folder')
@click.option("--walltime", default="01:30:00")
@click.option("--space_sample_size", default=2000)
@click.option("--request", default="select=1:ncpus=1:mem=8gb")
@click.option("--queue", default="HPCA-01839-EFR")
def pbs_indexer(graph_path, seed, n_jobs, results_folder, walltime, request, space_sample_size, queue):
    """
    Generates space_sample_size starting cut sets and divides them in to n jobs to be scheduled

    :param graph_path:
    :param seed:
    :param n_jobs:
    :param results_folder:
    :param walltime:
    :param request:
    :param space_sample_size:
    :param queue:
    :return:
    """
    random.seed(seed)
    graph = load_graph(graph_path)

    job_options = dict(
        walltime=walltime,
        results_folder=os.path.abspath(results_folder),
        request=request,
        queue=queue,
        graph_path=os.path.abspath(graph_path),
        n_jobs=n_jobs,
        n_samps=int(round(int(space_sample_size) / int(n_jobs)))
    )

    job_template = """#!/bin/bash
    #PBS -k oe
    #PBS -l {request}
    #PBS -l {walltime}
    #PBS -P {queue}

    JOB=$PBS_ARRAY_INDEX

    cd $WORK_DIR
    modindexer dist_partitions {graph_path} $JOB {n_jobs} {n_samps} {results_folder}

    """.format(**job_options)

    command_file = "get_index_{}-{}-{}.sh".format(graph.name, seed, space_sample_size)

    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    with open(command_file, "w+") as cf:
        cf.write(job_template)

    cmd_options = dict(
        command_file=command_file,
        n_jobs=n_jobs,
        options="",
    )
    # Write script to disk
    # print/run command for exec script
    cmd = "qsub {options} -J 1-{n_jobs} {command_file}".format(**cmd_options)
    click.echo("Run the following command:")
    click.echo(cmd)


@click.command()
@click.argument('graph_path')
@click.argument('job_id', type=int)
@click.argument('n_samps', type=int)
@click.argument('results_folder')
def dist_partitions(graph_path, job_id, n_samps, results_folder):
    """
    Distributed partition job task
    :return:
    """
    graph = load_graph(graph_path)

    results_file = os.path.join(os.path.abspath(results_folder), '{}-{}-cs.res'.format(graph.name, job_id))

    # We want unique starting cut sets for this job_id
    random.seed(job_id)
    for cs in range(n_samps):
        _, local_optima = gen_local_optima_community(graph)
        cut_set = partition_to_cut_set(graph, local_optima)

        # Append new cut sets to file individually. Slower IO but saves memory
        with open(results_file, "a") as rf:
            rf.write(json.dumps(cut_set))
            rf.write('\n')


@click.command()
@click.argument('graph_path')
@click.argument('results_folder')
@click.option('--cache_path', default='.ctq_cache')
def merge(graph_path, results_folder, cache_path):
    graph = load_graph(graph_path)
    results = []

    hashes = []

    for fp in glob.glob('{}/{}-*-cs.res'.format(results_folder, graph.name)):
        with open(fp) as res_file:
            for line in res_file:
                cut_set = json.loads(line.strip())
                hashobj = sha256(str(cut_set).encode('utf-8'))

                if hashobj.hexdigest() not in hashes:
                    par = partition_from_cut(graph, cut_set)
                    results.append(par)
                    hashes.append(hashobj.hexdigest())

    if not os.path.exists(cache_path):
        os.mkdir(cache_path)

    file_path = os.path.join(cache_path, '{}-dist_res.json.xz'.format(graph.name))
    # Dump as compressed json file
    dump(results, file_path)


@click.group()
def cli():
    """
    modindexer command
    :return:
    """
    pass


cli.add_command(pbs_indexer)
cli.add_command(dist_partitions)
cli.add_command(merge)
