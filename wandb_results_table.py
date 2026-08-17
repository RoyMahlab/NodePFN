"""Pull experiment results from wandb and print an exp_name x dataset table.

Each run logs two wandb Tables:
  * ``baseline_summary``   -- one row per classification dataset, incl. ``test_acc_mean``
  * ``regression_summary`` -- one row per (dataset, target), incl. ``test_mae_mean``

This script collects both for a list of runs (by name) and renders a single table
whose rows are experiments and whose columns are datasets. A classification
dataset shows ``test_acc_mean`` plus ``test_ap_mean`` where the baseline logged
one; a regression dataset shows a single ``test_mae_mean`` averaged over its
targets (``--per-target`` keeps them as separate ``dataset/target`` columns).

Example:
    uv run python wandb_results_table.py                       # default run list
    uv run python wandb_results_table.py --csv results.csv
    uv run python wandb_results_table.py --runs baseline_short_updated
"""

import argparse
import json
import os
import tempfile

import wandb

DEFAULT_RUNS = [
    'configs/geo_baseline.json',
    # 'configs/geo_baseline_low_gamma_dist.json',
    # 'configs/geo_baseline_mid_gamma_dist.json',
    # 'configs/geo_baseline_high_gamma_dist.json',
    # 'configs/geo_baseline_less_features.json',
    # 'configs/geo_baseline_less_geo_features.json',
    'full_baseline_1_gpu_batch_8',
]

ACC_TABLE = 'baseline_summary'
MAE_TABLE = 'regression_summary'


def load_table(run, key, download_dir):
    """Return the logged wandb Table `key` as (columns, rows), or None."""
    # wandb returns a SummarySubDict here, not a plain dict -- duck-type it
    entry = run.summary.get(key)
    try:
        file_path = entry['path']
    except (TypeError, KeyError):
        return None
    try:
        path = run.file(file_path).download(
            root=os.path.join(download_dir, run.id), replace=True)
    except Exception as exc:  # noqa: BLE001 -- keep going on partial runs
        print(f"  ! could not download {key} for {run.name}: {exc}")
        return None
    with open(path.name) as fh:
        data = json.load(fh)
    return data['columns'], data['data']


def rows_as_dicts(table):
    columns, rows = table
    return [dict(zip(columns, row)) for row in rows]


def collect_run(run, download_dir, per_target=False):
    """Return {(dataset_label, metric): value} for one run.

    A classification dataset contributes an ``acc`` entry plus, when the
    baseline logged one, an ``ap`` entry (``test_ap_mean`` is null for the
    multi-class datasets).
    """
    values = {}

    acc = load_table(run, ACC_TABLE, download_dir)
    if acc:
        for row in rows_as_dicts(acc):
            dataset = row.get('dataset')
            if dataset in (None, 'MEAN'):  # last row is the aggregate
                continue
            values[(dataset, 'acc')] = row.get('test_acc_mean')
            if row.get('test_ap_mean') is not None:
                values[(dataset, 'ap')] = row['test_ap_mean']
    else:
        # fall back to the per-dataset summary keys written alongside the table
        # (accuracy only -- test_ap_mean lives in the table and nowhere else)
        for key, val in run.summary.items():
            if key.startswith('test_acc/'):
                values[(key.split('/', 1)[1], 'acc')] = val

    mae = load_table(run, MAE_TABLE, download_dir)
    if mae:
        by_dataset = {}
        for row in rows_as_dicts(mae):
            dataset, target = row.get('dataset'), row.get('target')
            if dataset in (None, 'MEAN') or row.get('test_mae_mean') is None:
                continue
            if per_target:
                label = f"{dataset}/{target}" if target else dataset
                values[(label, 'mae')] = row['test_mae_mean']
            else:
                by_dataset.setdefault(dataset, []).append(row['test_mae_mean'])
        # one MAE per dataset: unweighted mean over its targets
        for dataset, maes in by_dataset.items():
            values[(dataset, 'mae')] = sum(maes) / len(maes)

    return values


def pick_runs(api, entity, project, names, keep_all, states):
    """Resolve run names to run objects, newest first within a name."""
    path = f"{entity}/{project}"
    found = {}
    for run in api.runs(path):
        if run.name in names and (not states or run.state in states):
            found.setdefault(run.name, []).append(run)

    selected = []
    for name in names:
        runs = sorted(found.get(name, []), key=lambda r: r.created_at, reverse=True)
        if not runs:
            print(f"  ! no run found for {name!r}")
            continue
        chosen = runs if keep_all else runs[:1]
        for run in chosen:
            label = name if len(chosen) == 1 else f"{name} [{run.id}]"
            selected.append((label, run))
        if len(runs) > 1 and not keep_all:
            print(f"  note: {name!r} has {len(runs)} runs, using newest "
                  f"({runs[0].id}, {runs[0].created_at}); pass --all-runs for the rest")
    return selected


def fmt(value, metric):
    if value is None:
        return '-'
    return f"{value:.4f}" if metric == 'mae' else f"{value * 100:.2f}"


def render_markdown(exp_labels, columns, table):
    header = ['experiment'] + [f"{dataset} ({metric})" for dataset, metric in columns]
    lines = ['| ' + ' | '.join(header) + ' |',
             '|' + '|'.join(['---'] * len(header)) + '|']
    for label in exp_labels:
        cells = [fmt(table[label].get(c), c[1]) for c in columns]
        lines.append('| ' + ' | '.join([label] + cells) + ' |')
    return '\n'.join(lines)


def write_csv(path, exp_labels, columns, table):
    import csv
    with open(path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['experiment'] + [dataset for dataset, _ in columns])
        writer.writerow(['metric'] + [metric for _, metric in columns])
        for label in exp_labels:
            writer.writerow([label] + [table[label].get(c) for c in columns])
    print(f"\nWrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--entity', default=None,
                        help='wandb entity (default: your default entity)')
    parser.add_argument('--project', default='NodePFN')
    parser.add_argument('--runs', nargs='+', default=DEFAULT_RUNS,
                        help='wandb run names (default: the geo_baseline configs '
                             '+ baseline_short_updated)')
    parser.add_argument('--all-runs', action='store_true',
                        help='keep every run matching a name instead of only the newest')
    parser.add_argument('--state', nargs='*', default=['finished'],
                        help="only consider runs in these states (default: finished; "
                             "pass --state with no values to allow any state)")
    parser.add_argument('--per-target', action='store_true',
                        help='give each regression target its own column instead of '
                             'averaging the targets into one MAE per dataset')
    parser.add_argument('--csv', default=None, help='also write the table to this CSV path')
    args = parser.parse_args()

    api = wandb.Api()
    entity = args.entity or api.default_entity
    print(f"Fetching runs from {entity}/{args.project}")

    selected = pick_runs(api, entity, args.project, args.runs, args.all_runs, set(args.state))
    if not selected:
        raise SystemExit('No matching runs found.')

    table, columns = {}, []
    with tempfile.TemporaryDirectory() as download_dir:
        for label, run in selected:
            print(f"  - {label} ({run.id})")
            values = collect_run(run, download_dir, per_target=args.per_target)
            table[label] = values
            columns += [c for c in values if c not in columns]

    # classification datasets first (acc then ap per dataset), then regression
    order = {'acc': 0, 'ap': 1, 'mae': 2}
    columns.sort(key=lambda c: (order[c[1]] == 2, c[0], order[c[1]]))
    exp_labels = [label for label, _ in selected]

    mae_note = 'per target' if args.per_target else 'mean over the dataset targets'
    print(f'\nacc = test_acc_mean (%), ap = test_ap_mean (%), '
          f'mae = test_mae_mean ({mae_note})\n')
    print(render_markdown(exp_labels, columns, table))

    if args.csv:
        write_csv(args.csv, exp_labels, columns, table)


if __name__ == '__main__':
    main()
