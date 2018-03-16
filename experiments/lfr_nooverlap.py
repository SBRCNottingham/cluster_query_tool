"""
HPC script for running evaluation of performance on benchmark graphs

This script is supposed to be run by the jobscript
"""
from experiments import get_benchmark, get_auc_scores_community
import numpy as np
import click
from subprocess import call
import fasteners
import json
import os


@fasteners.interprocess_locked('.ctq_lfr_bm_results_db.lock')
def save_results(value, results_path, *keys):
    """
    Handle the locking of shared results dicts
    """
    if not os.path.exists(results_path):
        with open(results_path, "w+") as rfile:
            json.dump({}, rfile)

    with open(results_path) as rfile:
        db = json.load(rfile)
        xv = db
        for key in keys[:-1]:
            print(key)
            try:
                xv = xv[str(key)]
            except KeyError:
                xv[str(key)] = {}
                xv = xv[str(key)]

        xv[str(keys[-1])] = value

    with open(results_path, "w+") as rfile:
        json.dump(db, rfile, indent=4)


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
@click.argument("results_file")
def auc_compute(seed, n, mu, results_file):
    pset = _base_params.copy()
    pset['n'] = int(n)
    pset['mu'] = float(mu)
    pset['seed'] = int(seed)

    graph, communities, index = get_benchmark(pset)
    results = []
    for c, comm in communities.items():
        for seed_size in _seed_sizes:
            if len(comm) > seed_size:
                # Seed of AUC scores for node this size
                auc_s = get_auc_scores_community(seed_size, comm, graph, index)
                results.append([seed, c, seed_size, len(comm), np.mean(auc_s), np.std(auc_s)])

    save_results(results, results_file, mu)


@click.command()
@click.argument("n")
@click.option("--mu_steps", default=10)
@click.option("--network_samples", default=10)
@click.option("--walltime", default="01:30:00")
@click.option("--execute/--no_exec", default=False)
@click.option("--queue", default="HPCA-01839-EFR")
def run_jobs(n, mu_steps, network_samples, walltime, execute, queue):
    script_template = """#!/bin/bash
#PBS -k oe
#PBS -l {request}
#PBS -l {walltime}
#PBS -P {queue}
#PBS -e $HOME/ctq_run/error/error_{n}_{mu}.txt
#PBS -o $HOME/ctq_run/ouput/output_{n}_{mu}.txt

mkdir -p $HOME/ctq_run/output/
mkdir -p $HOME/ctq_run/error/

WORK_DIR={workdir}
RESULTS_DIR=
JOB=$PBS_ARRAY_INDEX

mkdir -p $RESULTS_DIR
cd $WORK_DIR
python experiments/lfr_nooverlap.py auc_compute {n} {mu:.2f} $JOB $RESULTS_DIR/{results_file}

"""
    results_file = "lfr_bm_{}.json".format(n)

    script_settings = dict(
        walltime="walltime={}".format(walltime),
        request="select=1:ncpus=16:mem=16gb",
        queue=queue,
        workdir="$HOME/repos/cluster_query_tool",
        results_file=results_file,
        n=n
    )

    cmd_opt = dict(
        options="",
        jcount=network_samples,
        n=n
    )
    cmd_template = "qsub {options} -J 1-{jcount} {command_file}"

    commands_path = os.path.abspath(os.path.dirname(__file__))

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
