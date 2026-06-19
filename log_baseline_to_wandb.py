#!/usr/bin/env python
"""Run the node-classification baselines and log them to a single wandb run.

Reads the dataset commands from ``run_baseline.sh`` (the single source of
truth), runs each one, collects its summary metrics, and logs everything to
ONE wandb run as a summary table (one row per dataset) plus per-dataset
scalar metrics.

Usage (from repo root):
    python log_baseline_to_wandb.py \
        --wandb_project NodePFN --wandb_run_name baseline-sweep

    # only a subset of datasets
    python log_baseline_to_wandb.py --datasets cora citeseer pubmed
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile

import wandb

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_commands(script_path):
    """Extract (dataset, arg_list) for each active node_classification call."""
    commands = []
    with open(script_path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            if 'node_classification' not in line or not line.startswith('python'):
                continue
            tokens = shlex.split(line)
            # find the script/module token, args follow it
            idx = next((i for i, t in enumerate(tokens) if 'node_classification' in t), None)
            if idx is None:
                continue
            args = tokens[idx + 1:]
            dataset = None
            for i, t in enumerate(args):
                if t == '--dataset' and i + 1 < len(args):
                    dataset = args[i + 1]
                elif t.startswith('--dataset='):
                    dataset = t.split('=', 1)[1]
            if dataset is None:
                continue
            commands.append((dataset, args))
    return commands


def override_base_model_path(args, new_path):
    """Replace any --base_model_path in the arg list with new_path."""
    out, skip = [], False
    for t in args:
        if skip:
            skip = False
            continue
        if t == '--base_model_path':
            skip = True  # also drop its value (next token)
            continue
        if t.startswith('--base_model_path='):
            continue
        out.append(t)
    return out + ['--base_model_path', new_path]


def run_one(dataset, args):
    """Run a single dataset and return its results dict (or None on failure)."""
    with tempfile.NamedTemporaryFile('r', suffix='.json', delete=False) as tmp:
        results_path = tmp.name
    try:
        cmd = [sys.executable, '-m', 'nodepfn.node_classification', *args,
               '--results_json', results_path]
        env = dict(os.environ)
        # node_classification's imports need nodepfn/ on the path
        nodepfn_dir = os.path.join(REPO_ROOT, 'nodepfn')
        env['PYTHONPATH'] = nodepfn_dir + os.pathsep + env.get('PYTHONPATH', '')
        print(f"\n>>> [{dataset}] {' '.join(shlex.quote(c) for c in cmd)}")
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
        if proc.returncode != 0:
            print(f"!!! [{dataset}] exited with code {proc.returncode}; skipping")
            return None
        with open(results_path) as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001 - keep the sweep going
        print(f"!!! [{dataset}] failed: {e}; skipping")
        return None
    finally:
        if os.path.exists(results_path):
            os.remove(results_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--script', default=os.path.join(REPO_ROOT, 'run_baseline.sh'),
                        help='shell script to read dataset commands from')
    parser.add_argument('--datasets', nargs='*', default=None,
                        help='only run these datasets (default: all in the script)')
    parser.add_argument('--wandb_project', default='NodePFN')
    parser.add_argument('--wandb_entity', default=None)
    parser.add_argument('--wandb_run_name', default='node-classification-baseline')
    parser.add_argument('--wandb_run_id', default=None,
                        help='join/append to an existing wandb run with this id '
                             '(the short code in the run URL) instead of creating a new one')
    parser.add_argument('--base_model_path', default=None,
                        help='override --base_model_path in every dataset command '
                             '(e.g. evaluate a freshly pretrained model)')
    parser.add_argument('--dry_run', action='store_true',
                        help='parse and print the commands without running or logging')
    args = parser.parse_args()

    commands = parse_commands(args.script)
    if args.datasets:
        wanted = set(args.datasets)
        commands = [(d, a) for d, a in commands if d in wanted]

    print(f"Found {len(commands)} dataset commands in {args.script}:")
    for dataset, _ in commands:
        print(f"  - {dataset}")
    if args.dry_run:
        return

    init_kwargs = dict(project=args.wandb_project, entity=args.wandb_entity,
                       job_type='baseline',
                       config={'script': args.script, 'n_datasets': len(commands),
                               'baseline_base_model_path': args.base_model_path})
    if args.wandb_run_id:
        # append to an existing run instead of starting a new one
        init_kwargs['id'] = args.wandb_run_id
        init_kwargs['resume'] = 'must'
    else:
        init_kwargs['name'] = args.wandb_run_name
    run = wandb.init(**init_kwargs)

    columns = ['dataset', 'runs', 'test_acc_mean', 'test_acc_std',
               'valid_acc_mean', 'valid_acc_std', 'test_rocauc_mean',
               'test_rocauc_std', 'fit_time_mean', 'smoothing_steps',
               'n_components', 'n_ensemble', 'dim_reduction']
    table = wandb.Table(columns=columns)

    completed = []
    for dataset, cmd_args in commands:
        if args.base_model_path:
            cmd_args = override_base_model_path(cmd_args, args.base_model_path)
        res = run_one(dataset, cmd_args)
        if res is None:
            continue
        completed.append(res)
        cfg = res.get('config', {})
        table.add_data(
            res['dataset'], res['runs'],
            res['test_acc_mean'], res['test_acc_std'],
            res['valid_acc_mean'], res['valid_acc_std'],
            res['test_rocauc_mean'], res['test_rocauc_std'],
            res['fit_time_mean'], cfg.get('smoothing_steps'),
            cfg.get('n_components'), cfg.get('n_ensemble'),
            cfg.get('dim_reduction'),
        )
        # per-dataset scalar metrics, easy to compare across the run
        run.summary[f'test_acc/{dataset}'] = res['test_acc_mean']
        run.summary[f'valid_acc/{dataset}'] = res['valid_acc_mean']
        if res['test_rocauc_mean'] is not None:
            run.summary[f'test_rocauc/{dataset}'] = res['test_rocauc_mean']

    if completed:
        def col_mean(key):
            vals = [r[key] for r in completed if r.get(key) is not None]
            return sum(vals) / len(vals) if vals else None

        mean_test_acc = col_mean('test_acc_mean')
        # append a final MEAN row averaging the numeric metric columns across datasets
        table.add_data(
            'MEAN', len(completed),
            mean_test_acc, col_mean('test_acc_std'),
            col_mean('valid_acc_mean'), col_mean('valid_acc_std'),
            col_mean('test_rocauc_mean'), col_mean('test_rocauc_std'),
            col_mean('fit_time_mean'), None, None, None, None,
        )
        run.summary['mean_test_acc'] = mean_test_acc
        run.summary['n_datasets_completed'] = len(completed)
        print(f"\nLogged {len(completed)}/{len(commands)} datasets. "
              f"Mean test acc = {mean_test_acc*100:.2f}")

    run.log({'baseline_summary': table})
    run.finish()


if __name__ == '__main__':
    main()
