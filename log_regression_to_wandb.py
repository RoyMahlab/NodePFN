#!/usr/bin/env python
"""Run the node-regression baselines (BlueSky) and log them to a wandb run.

Mirrors log_baseline_to_wandb.py's flow but for nodepfn.node_regression:
reads dataset commands from run_bluesky_baseline.sh (or any script with the
same command shape), runs each, and logs one row per (dataset, target) to
wandb -- since node_regression.py's --target all evaluates likes/replies/
reposts as three independent regressions per dataset command.

Usage (from repo root):
    python log_regression_to_wandb.py --wandb_run_id <id>   # append to an existing run
    python log_regression_to_wandb.py --datasets bluesky_quotes
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
    """Extract (dataset, arg_list) for each active node_regression call."""
    commands = []
    with open(script_path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            if 'node_regression' not in line or not line.startswith('python'):
                continue
            tokens = shlex.split(line)
            idx = next((i for i, t in enumerate(tokens) if 'node_regression' in t), None)
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
    out, skip = [], False
    for t in args:
        if skip:
            skip = False
            continue
        if t == '--base_model_path':
            skip = True
            continue
        if t.startswith('--base_model_path='):
            continue
        out.append(t)
    return out + ['--base_model_path', new_path]


def cap_n_bins(args, max_bins):
    """Lower any --n_bins above max_bins (the checkpoint's max_num_classes).

    A quantile bin is a class at inference time, so a command asking for more bins than
    the model's head is wide fails in NodePFNClassifier.fit. Commands already at or below
    the cap are left alone; a command with no --n_bins gets the cap made explicit.
    """
    out, i, seen = [], 0, False
    while i < len(args):
        t = args[i]
        if t == '--n_bins' and i + 1 < len(args):
            out += ['--n_bins', str(min(int(args[i + 1]), max_bins))]
            seen, i = True, i + 2
            continue
        if t.startswith('--n_bins='):
            out.append(f"--n_bins={min(int(t.split('=', 1)[1]), max_bins)}")
            seen, i = True, i + 1
            continue
        out.append(t)
        i += 1
    return out if seen else out + ['--n_bins', str(max_bins)]


def run_one(dataset, args):
    """Run a single dataset command and return its results dict (or None on failure)."""
    with tempfile.NamedTemporaryFile('r', suffix='.json', delete=False) as tmp:
        results_path = tmp.name
    try:
        cmd = [sys.executable, '-m', 'nodepfn.node_regression', *args,
               '--results_json', results_path]
        env = dict(os.environ)
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
    parser.add_argument('--script', nargs='*',
                        default=[os.path.join(REPO_ROOT, 'run_bluesky_baseline.sh')],
                        help='shell script(s) to read dataset commands from')
    parser.add_argument('--datasets', nargs='*', default=None,
                        help='only run these datasets (default: all in the script)')
    parser.add_argument('--wandb_project', default='NodePFN')
    parser.add_argument('--wandb_entity', default=None)
    parser.add_argument('--wandb_run_name', default='node-regression-baseline')
    parser.add_argument('--wandb_run_id', default=None,
                        help='join/append to an existing wandb run with this id '
                             'instead of creating a new one')
    parser.add_argument('--base_model_path', default=None,
                        help='override --base_model_path in every dataset command')
    parser.add_argument('--max_n_bins', type=int, default=None,
                        help="cap --n_bins in every dataset command at this value, normally the "
                             "checkpoint's max_num_classes -- a quantile bin is a class at "
                             'inference, so more bins than the head is wide cannot be predicted')
    parser.add_argument('--dry_run', action='store_true',
                        help='parse and print the commands without running or logging')
    args = parser.parse_args()

    commands = [cmd for script in args.script for cmd in parse_commands(script)]
    if args.datasets:
        wanted = set(args.datasets)
        commands = [(d, a) for d, a in commands if d in wanted]

    print(f"Found {len(commands)} dataset commands in {', '.join(args.script)}:")
    for dataset, _ in commands:
        print(f"  - {dataset}")
    if args.dry_run:
        return

    init_kwargs = dict(project=args.wandb_project, entity=args.wandb_entity,
                       job_type='regression-baseline',
                       config={'script': args.script, 'n_datasets': len(commands),
                               'baseline_base_model_path': args.base_model_path,
                               'baseline_max_n_bins': args.max_n_bins})
    if args.wandb_run_id:
        init_kwargs['id'] = args.wandb_run_id
        init_kwargs['resume'] = 'must'
    else:
        init_kwargs['name'] = args.wandb_run_name
    run = wandb.init(**init_kwargs)

    columns = ['dataset', 'target', 'runs', 'n_bins',
               'val_mse_mean', 'val_mae_mean', 'val_r2_mean', 'val_spearman_mean',
               'test_mse_mean', 'test_mae_mean', 'test_r2_mean', 'test_spearman_mean',
               'fit_time_mean']
    table = wandb.Table(columns=columns)

    completed = []  # list of (dataset, target, metrics_dict)
    for dataset, cmd_args in commands:
        if args.base_model_path:
            cmd_args = override_base_model_path(cmd_args, args.base_model_path)
        if args.max_n_bins:
            cmd_args = cap_n_bins(cmd_args, args.max_n_bins)
        res = run_one(dataset, cmd_args)
        if res is None:
            continue
        for target, metrics in res.get('targets', {}).items():
            completed.append((dataset, target, metrics))
            table.add_data(
                dataset, target, res['runs'], res['n_bins'],
                metrics['val_mse_mean'], metrics['val_mae_mean'],
                metrics['val_r2_mean'], metrics['val_spearman_mean'],
                metrics['test_mse_mean'], metrics['test_mae_mean'],
                metrics['test_r2_mean'], metrics['test_spearman_mean'],
                metrics['fit_time_mean'],
            )
            run.summary[f'test_r2/{dataset}/{target}'] = metrics['test_r2_mean']
            run.summary[f'test_mse/{dataset}/{target}'] = metrics['test_mse_mean']

    if completed:
        mean_test_r2 = sum(m['test_r2_mean'] for _, _, m in completed) / len(completed)
        table.add_data('MEAN', 'ALL', len(completed), None,
                       None, None, None, None,
                       None, None, mean_test_r2, None, None)
        run.summary['mean_test_r2'] = mean_test_r2
        run.summary['n_dataset_targets_completed'] = len(completed)
        print(f"\nLogged {len(completed)} (dataset, target) results. "
              f"Mean test R2 = {mean_test_r2:.4f}")

    run.log({'regression_summary': table})
    run.finish()


if __name__ == '__main__':
    main()
