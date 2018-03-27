"""
HPC script for running evaluation of performance on benchmark graphs

This script is supposed to be run by the jobscript
"""
from experiments import get_benchmark, get_auc_scores_community
import numpy as np
import click
from subprocess import call
import json
import os

_base_params = dict(
    average_degree=20,
    max_degree=50,
    minc_size=10,
    maxc_size=50,
)

_seed_sizes = [1, 3, 7, 15]




@click.command()
@click.argument("n")
@click.argument("mu")
@click.argument("seed")
@click.argument("results_folder")
def auc_compute(seed, n, mu, results_folder):
    pset = _base_params.copy()
    pset['n'] = int(n)
    pset['mu'] = float(mu)
    pset['seed'] = int(seed)

    rp = "{}_{}_{}.json".format(n, mu, seed)
    results_file = os.path.abspath(os.path.join(results_folder, rp))

    graph, communities, index = get_benchmark(pset)
    results = []
    for c, comm in communities.items():
        for seed_size in _seed_sizes:
            if len(comm) > seed_size:
                # Seed of AUC scores for node this size
                auc_s = get_auc_scores_community(seed_size, comm, graph, index)
                results.append([int(n), int(mu), int(seed), c, seed_size, len(comm), np.mean(auc_s), np.std(auc_s)])

        with open(results_file, "w+") as rf:
            json.dump(results, rf)


@click.command()
@click.argument("n")
@click.option("--mu_steps", default=10)
@click.option("--network_samples", default=10)
@click.option("--walltime", default="01:30:00")
@click.option("--request", default="select=1:ncpus=16:mem=31gb")
@click.option("--execute/--no_exec", default=False)
@click.option("--queue", default="HPCA-01839-EFR")
def run_jobs(n, mu_steps, network_samples, walltime, request, execute, queue):
    script_template = """#!/bin/bash
#PBS -k oe
#PBS -l {request}
#PBS -l {walltime}
#PBS -P {queue}

WORK_DIR={workdir}
JOB=$PBS_ARRAY_INDEX

cd $WORK_DIR
python {_file} auc_compute {n} {mu:.2f} $JOB {results_folder}

"""

    workdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    results_folder = os.path.join(workdir, 'hpc_results', "lfr_no_overlap_{}".format(n))
    print(results_folder)

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    script_settings = dict(
        walltime="walltime={}".format(walltime),
        request=request,
        queue=queue,
        workdir=workdir,
        results_folder=results_folder,
        n=n,
        _file=__file__,
    )

    cmd_opt = dict(
        options="",
        jcount=network_samples,
        n=n,
        e="$HOME/ctq_run/error/error_{}.txt".format(n),
        o="$HOME/ctq_run/output/output_{}.txt".format(n),
    )
    cmd_template = "qsub {options} -J 1-{jcount} {command_file}"

    commands_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'job_scripts'))
    if not os.path.exists(commands_path):
        os.mkdir(commands_path)

    for mu in np.linspace(0, 1, mu_steps):
        command_file_path = os.path.join(commands_path, "lfr_bm_{}_{:.2f}.sh".format(n, mu))

        script_settings["mu"] = mu
        script = script_template.format(**script_settings)

        with open(command_file_path, "w+") as cmdfile:
            cmdfile.write(script)

        params = cmd_opt.copy()
        params['mu'] = mu
        params['command_file'] = command_file_path
        command = cmd_template.format(**params)
        click.echo(command)
        if execute:
            call(command.split())


@click.group()
def cli():
    pass


if __name__ == "__main__":
    cli.add_command(run_jobs)
    cli.add_command(auc_compute)
    cli()
