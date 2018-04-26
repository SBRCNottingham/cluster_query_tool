import click
import os
import json
import glob
from hashlib import sha256
from .indexer import partition_to_cut_set, partition_from_cut, dump
from .louvain_consensus import gen_local_optima_community
import random
import networkx as nx
from subprocess import check_output


def load_graph(path, graph_name=None):
    graph = nx.read_edgelist(path)

    if graph_name is not None:
        graph.name = graph_name
    else:
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
@click.option('--graph_name', default=None)
@click.option('--cache_path', default='.ctq_cache')
@click.option('--cache_name', default=None)
def pbs_indexer(graph_path, seed, n_jobs, results_folder, walltime, request, space_sample_size, queue,
                graph_name, cache_path, cache_name):
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
    graph = load_graph(graph_path, graph_name)

    if cache_name is None:
        cache_name = '{}-dist_res.json.xz'.format(graph.name)

    job_options = dict(
        walltime=walltime,
        results_folder=os.path.abspath(results_folder),
        request=request,
        queue=queue,
        graph_path=os.path.abspath(graph_path),
        n_jobs=n_jobs,
        seed=seed,
        n_samps=int(round(int(space_sample_size) / int(n_jobs))),
        graph_name=graph.name,
        cache_path=os.path.abspath(cache_path),
        cache_name=cache_name,
    )

    job_template = """#!/bin/bash
#PBS -l {request}
#PBS -l walltime={walltime}
#PBS -P {queue}

JOB=$PBS_ARRAY_INDEX

modindexer dist_partitions {graph_path} $JOB {n_samps} {seed} {results_folder}

    """.format(**job_options)

    command_file = "get_index_{}-{}-{}.sh".format(graph.name, seed, space_sample_size)

    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    with open(command_file, "w+") as cf:
        cf.write(job_template)

    merge_template = """#!/bin/bash
#PBS -l {request}
#PBS -l walltime={walltime}
#PBS -P {queue}

modindexer merge {graph_path} {results_folder} --graph_name {graph_name} --cache_path {cache_path} --cache_name {cache_name}

    """.format(**job_options)

    merge_file = "merge_index_{}-{}-{}.sh".format(graph.name, seed, space_sample_size)
    with open(merge_file, "w+") as cf:
        cf.write(merge_template)

    cmd_options = dict(
        command_file=command_file,
        n_jobs=n_jobs,
        merge_file=merge_file,
        options="",
    )
    # Write script to disk
    # print/run command for exec script
    jcmd = "qsub {options} -J 1-{n_jobs} {command_file}".format(**cmd_options)

    click.echo("Running submission:")

    job_out = check_output(jcmd.split())
    click.echo(job_out)

    cmd_options["dist_proc"] = job_out.split()[0]
    mcmd = "qsub {options} {merge_file} -W depend=afterany:{dist_proc}".format(**cmd_options)
    j2 = check_output(mcmd.split())

    click.echo(j2)


@click.command()
@click.argument('graph_path')
@click.argument('job_id', type=int)
@click.argument('n_samps', type=int)
@click.argument('seed', type=int)
@click.argument('results_folder')
@click.option('--graph_name', default=None)
def dist_partitions(graph_path, job_id, n_samps, seed, results_folder, graph_name):
    """
    Distributed partition job task
    :return:
    """
    graph = load_graph(graph_path, graph_name)

    results_file = os.path.join(os.path.abspath(results_folder), '{}-{}-cs.res'.format(graph.name, job_id))

    # We want unique starting cut sets for this job_id
    random.seed(job_id + seed)
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
@click.option('--cache_name', default=None)
@click.option('--graph_name', default=None)
def merge(graph_path, results_folder, cache_path, cache_name, graph_name):
    """
    Merge results
    :param graph_path:
    :param results_folder:
    :param cache_path:
    :param cache_name:
    :param graph_name:
    :return:
    """
    graph = load_graph(graph_path, graph_name)

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

    if cache_name is None:
        cache_name = '{}-dist_res.json.xz'.format(graph.name)

    file_path = os.path.join(cache_path, cache_name)
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
