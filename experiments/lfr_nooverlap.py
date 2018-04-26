"""
HPC script for running evaluation of performance on benchmark graphs

This script is supposed to be run by the jobscript
"""
from experiments import get_benchmark, get_auc_scores_community
from cluster_query_tool.louvain_consensus import membership_matrix
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

    membership_ma, nmap = membership_matrix(graph.nodes(), index)

    nodes = np.array(list(nmap.values()))
    results = []
    for c, comm in communities.items():
        for seed_size in _seed_sizes:
            if len(comm) > seed_size:
                # Seed of AUC scores for node this size
                scom = np.array([nmap[i] for i in comm])
                auc_s = get_auc_scores_community(seed_size, scom, nodes, membership_ma)
                results.append([int(n), float(mu), int(seed), c, seed_size, len(comm), np.mean(auc_s), np.std(auc_s)])

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

    opt_path = os.path.join(workdir, 'hpc_out', 'output_{}'.format(n))
    err_path = os.path.join(workdir, 'hpc_out', 'error_{}'.format(n))

    os.makedirs(os.path.join(workdir, 'hpc_out'), exist_ok=True)

    cmd_opt = dict(
        options="",
        jcount=network_samples,
        n=n,
        e=err_path,
        o=opt_path,
    )
    cmd_template = "qsub {options} -J 1-{jcount} {command_file} -e {e} -o {o}"

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
        params['command_file'] = command_file_path
        command = cmd_template.format(**params)
        click.echo(command)
        if execute:
            call(command.split())


@click.command()
@click.argument("n")
def gen_figure(n):
    import json
    from matplotlib import pyplot as plt
    import glob
    import pandas as pd
    rows = []

    for f in glob.glob("hpc_results/lfr_no_overlap_{}/*.json".format(n)):
        with open(f) as jf:
            rts = json.load(jf)
            for row in rts:
                rows.append(row)

    df = pd.DataFrame(rows, columns=['n', 'mixing', 'seed', 'c', 'seed_size', 'comm', 'auc', 'auc_std'])

    fig, ax = plt.subplots()
    fig.set_dpi(90)
    ax.set_ylabel("Mean AUC")
    ax.set_xlabel("Mixing coefficient $\mu$")
    ax.set_ylim(0.4, 1.01)
    ax.set_xlim(0.0, 1.01)
    x_vals = df['mixing'].unique()
    x_vals.sort()

    for s in _seed_sizes:
        y_vals = []
        y_err = []
        sdf = df.loc[df['seed_size'] == s]
        for x in x_vals:
            m = sdf.loc[sdf['mixing'] == x]["auc"].mean()
            y_vals.append(m)
            std = sdf.loc[sdf['mixing'] == x]["auc_std"].std()
            y_err.append(std)
        ax.scatter(x_vals, y_vals, label="{} seed nodes".format(s))
        ax.errorbar(x_vals, y_vals, yerr=y_err)

    ax.legend(loc=3)
    fig.savefig("article/images/lfr_binary_mo_overlap_auc_{}.eps".format(n))
    fig.savefig("article/images/lfr_binary_mo_overlap_auc_{}.svg".format(n))
    fig.savefig("article/images/lfr_binary_mo_overlap_auc_{}.png".format(n))


@click.group()
def cli():
    pass


if __name__ == "__main__":
    cli.add_command(run_jobs)
    cli.add_command(auc_compute)
    cli.add_command(gen_figure)
    cli()
