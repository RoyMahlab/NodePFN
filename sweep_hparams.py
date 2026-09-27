#!/usr/bin/env python
"""Sweep {dim_reduction, n_components, smoothing_steps, n_ensemble} across checkpoints.

Reads the dataset commands out of run_extra_baseline.sh / run_bluesky_baseline.sh /
run_graphland_baseline.sh (same convention as log_baseline_to_wandb.py /
log_regression_to_wandb.py), then for every (checkpoint, hyperparam combo) re-runs
every dataset command with:
  * --base_model_path forced to the checkpoint under test
  * --runs / --query_batch_size / --batch_size_inference / --precision forced to the
    fixed values below (these are NOT swept)
  * --dim_reduction / --n_components / --smoothing_steps / --n_ensemble overridden to
    the combo under test
Jobs are dispatched across up to --n_gpus GPUs in parallel (each subprocess gets its
own CUDA_VISIBLE_DEVICES slice, so its own default --device 0 / --pipeline_gpus lands
on the assigned physical GPU(s)).

Each (checkpoint, combo) is logged as its own wandb run (job_type='hparam-sweep'),
tagged with the checkpoint label and combo, mirroring log_baseline_to_wandb.py's
per-dataset table. After every job finishes, a local cross-checkpoint comparison
table/CSV is written under --out_dir showing, per combo and per dataset, whether the
"geo" checkpoint beat the "full" checkpoint.

Usage (from repo root):
    # smoke test on one fast dataset, no wandb, small grid, before committing GPU time
    python sweep_hparams.py --datasets dblp --skip_wandb --dry_run
    python sweep_hparams.py --datasets dblp --skip_wandb

    # full curated-grid sweep across all 3 scripts, both checkpoints, 8-way parallel
    python sweep_hparams.py

    # widen the grid
    python sweep_hparams.py --smoothing_steps_values 0 1 2 3 --n_ensemble_values 4 8 16 32
"""
import argparse
import datetime
import json
import os
import shlex
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

MODULES = {
    'node_classification': 'nodepfn.node_classification',
    'node_regression': 'nodepfn.node_regression',
}

# params that are NEVER swept -- forced to these values on every job
FIXED_DEFAULTS = dict(runs=5, query_batch_size=2000, batch_size_inference=1, precision='bf16')


# --------------------------------------------------------------------------- #
# parsing dataset commands out of the shell scripts
# --------------------------------------------------------------------------- #

def parse_commands(script_path):
    """Extract {module, dataset, target, args} for each active node_* call in a script."""
    commands = []
    with open(script_path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#') or not line.startswith('python'):
                continue
            module = next((m for key, m in MODULES.items() if key in line), None)
            if module is None:
                continue
            tokens = shlex.split(line)
            idx = next((i for i, t in enumerate(tokens) if module.split('.')[-1] in t), None)
            if idx is None:
                continue
            args = tokens[idx + 1:]
            dataset = target = None
            for i, t in enumerate(args):
                if t == '--dataset' and i + 1 < len(args):
                    dataset = args[i + 1]
                elif t.startswith('--dataset='):
                    dataset = t.split('=', 1)[1]
                elif t == '--target' and i + 1 < len(args):
                    target = args[i + 1]
                elif t.startswith('--target='):
                    target = t.split('=', 1)[1]
            if dataset is None:
                continue
            commands.append({'module': module, 'dataset': dataset, 'target': target,
                              'args': args, 'script': script_path})
    return commands


def override_args(args, overrides):
    """Strip any existing --key(=value) for keys in `overrides` and append the new values.

    All overridden keys here take a value (none are store_true flags).
    """
    keys = {f'--{k}' for k in overrides}
    out, i = [], 0
    while i < len(args):
        t = args[i]
        bare = t.split('=', 1)[0]
        if bare in keys:
            i += 1 if '=' in t else 2
            continue
        out.append(t)
        i += 1
    for k, v in overrides.items():
        out += [f'--{k}', str(v)]
    return out


def parse_pipeline_gpus(args):
    for i, t in enumerate(args):
        if t == '--pipeline_gpus' and i + 1 < len(args):
            return int(args[i + 1])
        if t.startswith('--pipeline_gpus='):
            return int(t.split('=', 1)[1])
    return 1


# --------------------------------------------------------------------------- #
# hyperparameter grid
# --------------------------------------------------------------------------- #

def generate_grid(dim_reduction_values, n_components_values, smoothing_steps_values, n_ensemble_values):
    """Cartesian product, skipping n_components entirely when dim_reduction == 'none'
    (it's unused there, so sweeping it would just rerun identical jobs)."""
    grid = []
    for dr in dim_reduction_values:
        nc_values = n_components_values if dr == 'tsvd' else [None]
        for nc in nc_values:
            for ss in smoothing_steps_values:
                for ne in n_ensemble_values:
                    combo = {'dim_reduction': dr, 'n_components': nc,
                             'smoothing_steps': ss, 'n_ensemble': ne}
                    combo['key'] = combo_key(combo)
                    grid.append(combo)
    return grid


def combo_key(combo):
    parts = [f"dr={combo['dim_reduction']}"]
    if combo['n_components'] is not None:
        parts.append(f"nc={combo['n_components']}")
    parts += [f"ss={combo['smoothing_steps']}", f"ne={combo['n_ensemble']}"]
    return '_'.join(parts)


# --------------------------------------------------------------------------- #
# GPU pool + job execution
# --------------------------------------------------------------------------- #

class GPUPool:
    def __init__(self, n_gpus):
        self._cond = threading.Condition()
        self._free = list(range(n_gpus))

    def acquire(self, n):
        with self._cond:
            while len(self._free) < n:
                self._cond.wait()
            return [self._free.pop(0) for _ in range(n)]

    def release(self, devices):
        with self._cond:
            self._free.extend(devices)
            self._cond.notify_all()


def run_job(job, gpu_pool):
    """Run one subprocess job, return the parsed results_json dict (or None on failure)."""
    devices = gpu_pool.acquire(job['n_gpus_needed'])
    try:
        with tempfile.NamedTemporaryFile('r', suffix='.json', delete=False) as tmp:
            results_path = tmp.name
        try:
            cmd = [sys.executable, '-m', job['module'], *job['args'], '--results_json', results_path]
            env = dict(os.environ)
            env['CUDA_VISIBLE_DEVICES'] = ','.join(str(d) for d in devices)
            env['PYTHONPATH'] = os.path.join(REPO_ROOT, 'nodepfn') + os.pathsep + env.get('PYTHONPATH', '')
            label = f"{job['checkpoint_label']}/{job['combo_key']}/{job['dataset']}"
            print(f">>> [gpu {devices}] [{label}] {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
            proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
            if proc.returncode != 0:
                print(f"!!! [{label}] exited with code {proc.returncode}; skipping", flush=True)
                return None
            with open(results_path) as f:
                return json.load(f)
        except Exception as e:  # noqa: BLE001 - keep the sweep going
            print(f"!!! [{job['dataset']}] failed: {e}; skipping", flush=True)
            return None
        finally:
            if os.path.exists(results_path):
                os.remove(results_path)
    finally:
        gpu_pool.release(devices)


# --------------------------------------------------------------------------- #
# result -> row normalization (classification and regression share one schema)
# --------------------------------------------------------------------------- #

def result_rows(module, dataset, result):
    """Flatten one job's results_json into rows of a common shape, one per
    (dataset, target) -- target is '' for classification."""
    rows = []
    if result is None:
        return rows
    if module == MODULES['node_classification']:
        has_ap = result.get('test_ap_mean') is not None
        rows.append({
            'dataset': dataset, 'target': '', 'task': 'classification',
            'metric_name': 'ap' if has_ap else 'acc',
            'metric_value': result['test_ap_mean'] if has_ap else result['test_acc_mean'],
            'test_acc_mean': result.get('test_acc_mean'),
            'test_rocauc_mean': result.get('test_rocauc_mean'),
            'test_ap_mean': result.get('test_ap_mean'),
            'test_r2_mean': None, 'test_mae_mean': None,
            'fit_time_mean': result.get('fit_time_mean'), 'runs': result.get('runs'),
        })
    else:
        for target, metrics in result.get('targets', {}).items():
            rows.append({
                'dataset': dataset, 'target': target, 'task': 'regression',
                'metric_name': 'r2', 'metric_value': metrics.get('test_r2_mean'),
                'test_acc_mean': None, 'test_rocauc_mean': None, 'test_ap_mean': None,
                'test_r2_mean': metrics.get('test_r2_mean'), 'test_mae_mean': metrics.get('test_mae_mean'),
                'fit_time_mean': metrics.get('fit_time_mean'), 'runs': result.get('runs'),
            })
    return rows


def log_group_to_wandb(project, entity, run_name_prefix, checkpoint_label, checkpoint_path,
                        combo, fixed, rows):
    import wandb
    tags = [checkpoint_label] + [f"{k}={v}" for k, v in combo.items() if k != 'key' and v is not None]
    run = wandb.init(project=project, entity=entity, job_type='hparam-sweep',
                      name=f"{run_name_prefix}_{checkpoint_label}_{combo['key']}",
                      tags=tags,
                      config={'checkpoint_label': checkpoint_label, 'checkpoint_path': checkpoint_path,
                              **{k: v for k, v in combo.items() if k != 'key'}, **fixed})
    columns = ['dataset', 'target', 'task', 'metric_name', 'metric_value',
               'test_acc_mean', 'test_rocauc_mean', 'test_ap_mean', 'test_r2_mean',
               'test_mae_mean', 'fit_time_mean', 'runs']
    table = wandb.Table(columns=columns)
    for row in rows:
        table.add_data(*[row[c] for c in columns])
        label = f"{row['dataset']}/{row['target']}" if row['target'] else row['dataset']
        run.summary[f"{row['metric_name']}/{label}"] = row['metric_value']

    cls_vals = [r['metric_value'] for r in rows if r['task'] == 'classification' and r['metric_value'] is not None]
    reg_vals = [r['metric_value'] for r in rows if r['task'] == 'regression' and r['metric_value'] is not None]
    if cls_vals:
        run.summary['mean_classification_metric'] = sum(cls_vals) / len(cls_vals)
    if reg_vals:
        run.summary['mean_r2'] = sum(reg_vals) / len(reg_vals)
    run.summary['n_rows_completed'] = len(rows)

    run.log({'sweep_summary': table})
    run.finish()


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scripts', nargs='+', default=[
        # os.path.join(REPO_ROOT, 'run_extra_baseline.sh'),
        # os.path.join(REPO_ROOT, 'run_bluesky_baseline.sh'),
        os.path.join(REPO_ROOT, 'run_graphland_baseline.sh'),
    ])
    parser.add_argument('--datasets', nargs='*', default=None,
                        help='only run these datasets (default: all found in --scripts)')

    parser.add_argument('--checkpoints', nargs='+', default=[
        'models_ckpts/full_baseline_8_gpus',
        'models_ckpts/geo_baseline_less_features_less_layers_8_gpus_uniform_geo_prior',
        'models_ckpts/geo_baseline_8_gpus_uniform_geo_prior',
        'models_ckpts/geo_baseline_less_features_8_gpus',
        'models_ckpts/geo_baseline_only_zscore_norm_8_gpus'

    ])
    parser.add_argument('--checkpoint_labels', nargs='+', default=[
                    'full',
                    'geo',
                    'geo_baseline_8_gpus_uniform_geo_prior', 
                    'geo_baseline_less_features_8_gpus', 
                    'geo_baseline_only_zscore_norm_8_gpus'
                ],
        help='short labels for --checkpoints, same order/length')

    parser.add_argument('--dim_reduction_values', nargs='+', default=['none', 'tsvd'], choices=['none', 'tsvd'])
    parser.add_argument('--n_components_values', nargs='+', type=int, default=[8, 16, 32],
                        help='only used for dim_reduction=tsvd combos')
    parser.add_argument('--smoothing_steps_values', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument('--n_ensemble_values', nargs='+', type=int, default=[2, 4, 8, 16])

    parser.add_argument('--runs', type=int, default=FIXED_DEFAULTS['runs'])
    parser.add_argument('--query_batch_size', type=int, default=FIXED_DEFAULTS['query_batch_size'])
    parser.add_argument('--batch_size_inference', type=int, default=FIXED_DEFAULTS['batch_size_inference'])
    parser.add_argument('--precision', default=FIXED_DEFAULTS['precision'], choices=['fp32', 'fp16', 'bf16'])

    parser.add_argument('--n_gpus', type=int, default=8, help='max concurrent jobs / GPUs to use')

    parser.add_argument('--wandb_project', default='NodePFN-sweep-short')
    parser.add_argument('--wandb_entity', default=None)
    parser.add_argument('--run_name_prefix', default='hsweep')
    parser.add_argument('--skip_wandb', action='store_true', help='do not log to wandb (local-only smoke test)')

    parser.add_argument('--out_dir', default=None,
                        help='where to write the local summary CSV/JSON (default: sweep_results/<timestamp>/)')
    parser.add_argument('--dry_run', action='store_true',
                        help='print the planned jobs/grid without running or logging anything')
    args = parser.parse_args()

    if len(args.checkpoints) != len(args.checkpoint_labels):
        raise SystemExit('--checkpoints and --checkpoint_labels must have the same length')

    fixed = {'runs': args.runs, 'query_batch_size': args.query_batch_size,
             'batch_size_inference': args.batch_size_inference, 'precision': args.precision}

    commands = [cmd for script in args.scripts for cmd in parse_commands(script)]
    if args.datasets:
        wanted = set(args.datasets)
        commands = [c for c in commands if c['dataset'] in wanted]
    if not commands:
        raise SystemExit('No dataset commands found (check --scripts / --datasets).')

    grid = generate_grid(args.dim_reduction_values, args.n_components_values,
                          args.smoothing_steps_values, args.n_ensemble_values)
    print(f"{len(commands)} dataset command(s): {[c['dataset'] for c in commands]}")
    print(f"{len(grid)} hyperparam combo(s): {[g['key'] for g in grid]}")
    print(f"{len(args.checkpoints)} checkpoint(s): {list(zip(args.checkpoint_labels, args.checkpoints))}")
    total_jobs = len(commands) * len(grid) * len(args.checkpoints)
    print(f"=> {total_jobs} subprocess job(s) total, up to {args.n_gpus} in parallel\n")
    # build the full job list
    jobs = []
    for label, path in zip(args.checkpoint_labels, args.checkpoints):
        for combo in grid:
            overrides = dict(base_model_path=path, **fixed,
                              dim_reduction=combo['dim_reduction'], smoothing_steps=combo['smoothing_steps'],
                              n_ensemble=combo['n_ensemble'])
            if combo['n_components'] is not None:
                overrides['n_components'] = combo['n_components']
            for cmd in commands:
                final_args = override_args(cmd['args'], overrides)
                jobs.append({
                    'checkpoint_label': label, 'checkpoint_path': path,
                    'combo': combo, 'combo_key': combo['key'],
                    'module': cmd['module'], 'dataset': cmd['dataset'], 'target': cmd['target'],
                    'args': final_args, 'n_gpus_needed': parse_pipeline_gpus(final_args),
                })

    if args.dry_run:
        for j in jobs[:20]:
            print(f"  [{j['checkpoint_label']}/{j['combo_key']}] {j['dataset']}: "
                  f"python -m {j['module']} {' '.join(shlex.quote(a) for a in j['args'])}")
        if len(jobs) > 20:
            print(f"  ... and {len(jobs) - 20} more")
        return

    out_dir = args.out_dir or os.path.join(
        REPO_ROOT, 'sweep_results', datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(out_dir, exist_ok=True)

    # group jobs by (checkpoint_label, combo_key) so each group becomes one wandb run
    group_expected, group_rows, group_lock = {}, {}, threading.Lock()
    for j in jobs:
        gk = (j['checkpoint_label'], j['combo_key'])
        group_expected[gk] = group_expected.get(gk, 0) + 1
        group_rows.setdefault(gk, [])

    raw_results = []  # every individual job result, for the local JSON dump
    gpu_pool = GPUPool(args.n_gpus)
    progress_path = os.path.join(out_dir, 'progress.json')
    progress = {'total': total_jobs, 'completed': 0, 'succeeded': 0, 'failed': 0,
                'started_at': datetime.datetime.now().isoformat(), 'out_dir': out_dir}

    def write_progress():
        with open(progress_path, 'w') as f:
            json.dump({**progress, 'updated_at': datetime.datetime.now().isoformat()}, f, indent=2)

    write_progress()

    def finalize_group(gk):
        label, ck = gk
        combo = next(g for g in grid if g['key'] == ck)
        path = next(p for l, p in zip(args.checkpoint_labels, args.checkpoints) if l == label)
        rows = group_rows[gk]
        if not rows:
            logger.error(f"!!! [{label}/{ck}] no successful jobs; not logging to wandb")
            return
        if not args.skip_wandb:
            log_group_to_wandb(args.wandb_project, args.wandb_entity, args.run_name_prefix,
                                label, path, combo, fixed, rows)
        logger.info(f"=== finalized [{label}/{ck}]: {len(rows)} row(s) logged")

    with ThreadPoolExecutor(max_workers=args.n_gpus) as pool:
        futures = {pool.submit(run_job, j, gpu_pool): j for j in jobs}
        for fut in as_completed(futures):
            j = futures[fut]
            result = fut.result()
            raw_results.append({'checkpoint_label': j['checkpoint_label'], 'combo_key': j['combo_key'],
                                 'dataset': j['dataset'], 'module': j['module'], 'result': result})
            rows = result_rows(j['module'], j['dataset'], result)
            gk = (j['checkpoint_label'], j['combo_key'])
            with group_lock:
                group_rows[gk].extend(rows)
                group_expected[gk] -= 1
                done = group_expected[gk] == 0
                progress['completed'] += 1
                progress['succeeded' if result is not None else 'failed'] += 1
                completed_snapshot = progress['completed']
                write_progress()
            status = 'ok' if result is not None else 'FAILED'
            logger.info(f"[{completed_snapshot}/{total_jobs}] {status}: "
                  f"{j['checkpoint_label']}/{j['combo_key']}/{j['dataset']}", flush=True)
            if done:
                finalize_group(gk)

    with open(os.path.join(out_dir, 'raw_results.json'), 'w') as f:
        json.dump(raw_results, f, indent=2)

    write_comparison(out_dir, grid, args.checkpoint_labels, group_rows)
    print(f"\nWrote raw results and comparison summary to {out_dir}")


def write_comparison(out_dir, grid, checkpoint_labels, group_rows):
    """Per combo, per (dataset,target): does the 2nd checkpoint label beat the 1st?"""
    if len(checkpoint_labels) < 2:
        return
    base_label, other_label = checkpoint_labels[0], checkpoint_labels[1]

    import csv
    csv_path = os.path.join(out_dir, 'comparison.csv')
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['combo', 'dataset', 'target', 'metric', base_label, other_label,
                         f'{other_label}_minus_{base_label}', f'{other_label}_wins'])
        for combo in grid:
            ck = combo['key']
            base_rows = {(r['dataset'], r['target']): r for r in group_rows.get((base_label, ck), [])}
            other_rows = {(r['dataset'], r['target']): r for r in group_rows.get((other_label, ck), [])}
            keys = sorted(set(base_rows) | set(other_rows))
            wins = total = 0
            for dt in keys:
                br, orow = base_rows.get(dt), other_rows.get(dt)
                bv = br['metric_value'] if br else None
                ov = orow['metric_value'] if orow else None
                metric = (br or orow)['metric_name']
                diff = (ov - bv) if (bv is not None and ov is not None) else None
                win = diff is not None and diff > 0
                if diff is not None:
                    total += 1
                    wins += int(win)
                writer.writerow([ck, dt[0], dt[1], metric, bv, ov, diff, win])
            if total:
                logger.info(f"  [{ck}] {other_label} beats {base_label} on {wins}/{total} dataset(s)")
    logger.info(f"Wrote {csv_path}")


if __name__ == '__main__':
    main()

