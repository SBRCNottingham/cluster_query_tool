"""
HPC script for running evaluation of performance on benchmark graphs

This script is supposed to be run by the jobscript
"""
import experiments as exps
import numpy as np
import click
from subprocess import call

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
def auc_compute(seed, n, results_file, mu):
    click.echo("Starting")
    pset = _base_params.copy()
    pset['n'] = int(n)
    pset['mu'] = float(mu)
    pset['seed'] = int(seed)

    graph, communities, index = exps.get_benchmark(pset)
    results = {}
    for c, comm in communities.items():
        results[str(c)] = {}
        for seed_size in _seed_sizes:
            if len(comm) > seed_size:
                # Seed of AUC scores for node this size
                results[str(c)][str(seed_size)] = exps.get_auc_scores_community(seed_size, comm, graph, index)

    exps.save_results(results, results_file, mu, seed)


@click.command()
@click.argument("n")
@click.argument("mu")
@click.argument("seed")
def get_bm(n, mu, seed):
    pset = _base_params.copy()
    pset['n'] = int(n)
    pset['mu'] = float(mu)
    pset['seed'] = int(seed)
    # generate the cache of search indexes
    exps.get_benchmark(pset)


@click.command()
@click.argument("n")
@click.option("--mu_steps", default=10)
@click.option("--network_samples", default=10)
@click.option("--exec/--no_exec", default=False)
def print_commands(n, mu_steps, network_samples, exec):

    opt = dict(
        options="-l walltime=0:30:00 -l select=1:ncpus=16:mem=16gb",
        jcount=network_samples,
        n=n
    )

    template = "qsub {options} -J 1-{jcount} experiments/lfr_no_overlap_benchmarks.sh {n} {mu:.2f} "
    template2 = "qsub {options} -J 1-{jcount} experiments/lfr_no_overlap.sh {n} {mu:.2f}"
    for mu in np.linspace(0, 1, mu_steps):
        params = opt.copy()
        params['mu'] = mu
        command = template.format(**params)
        click.echo(command)

        if exec:
            call(command.split())

    for mu in np.linspace(0, 1, mu_steps):
        params = opt.copy()
        params['mu'] = mu
        command = template2.format(**params)
        click.echo(command)
        if exec:
            call(command.split())


@click.group()
def cli():
    pass


if __name__ == "__main__":
    cli.add_command(print_commands)
    cli.add_command(auc_compute)
    cli.add_command(get_bm)
    cli()
