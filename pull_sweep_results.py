#!/usr/bin/env python
"""Pull sweep_hparams.py results from wandb and rank hyperparam combos per checkpoint.

Each sweep_hparams.py (checkpoint, combo) run logs job_type='hparam-sweep' with a
'sweep_summary' table (one row per dataset/target) plus per-dataset summary scalars
and a 'mean_classification_metric' / 'mean_r2' rollup. This script fetches every such
run in a project, reconstructs one row per (checkpoint_label, combo), and writes two
ranked tables: the best combos for the baseline checkpoint and the best combos for the
"geo" checkpoint, plus the full per-dataset breakdown as CSV.

Safe to re-run any time -- a sweep still in progress just yields partial rankings.

Usage (from repo root):
    python pull_sweep_results.py
    python pull_sweep_results.py --project NodePFN-sweep --out_dir sweep_results
"""
import argparse
import json
import os
import tempfile

from tqdm import tqdm
import wandb

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def combo_key_from_config(cfg):
    parts = [f"dr={cfg.get('dim_reduction')}"]
    if cfg.get('dim_reduction') == 'tsvd' and cfg.get('n_components') is not None:
        parts.append(f"nc={cfg.get('n_components')}")
    parts += [f"ss={cfg.get('smoothing_steps')}", f"ne={cfg.get('n_ensemble')}"]
    return '_'.join(parts)


def load_sweep_table(run, download_dir):
    entry = run.summary.get('sweep_summary')
    try:
        file_path = entry['path']
    except (TypeError, KeyError):
        return None
    try:
        path = run.file(file_path).download(root=os.path.join(download_dir, run.id), replace=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  ! could not download sweep_summary for {run.name}: {exc}")
        return None
    with open(path.name) as fh:
        data = json.load(fh)
    columns, rows = data['columns'], data['data']
    return [dict(zip(columns, row)) for row in rows]


def collect_runs(entity, project):
    api = wandb.Api()
    entity = entity or api.default_entity
    # api.runs() returns lightweight run objects whose .config comes back empty;
    # re-fetching each by id via api.run() gives the fully populated config.
    run_ids = [r.id for r in tqdm(api.runs(f'{entity}/{project}'), desc="Fetching runs") if r.job_type == 'hparam-sweep']
    print(f"Found {len(run_ids)} hparam-sweep run(s) in {entity}/{project}")

    records = []
    with tempfile.TemporaryDirectory() as download_dir:
        for run_id in tqdm(run_ids, desc="Processing runs"):
            run = api.run(f'{entity}/{project}/{run_id}')
            cfg = run.config
            label = cfg.get('checkpoint_label', 'unknown')
            combo_key = combo_key_from_config(cfg)
            per_dataset_rows = load_sweep_table(run, download_dir) or []
            record = {
                'checkpoint_label': label,
                'checkpoint_path': cfg.get('checkpoint_path'),
                'combo': combo_key,
                'dim_reduction': cfg.get('dim_reduction'),
                'n_components': cfg.get('n_components'),
                'smoothing_steps': cfg.get('smoothing_steps'),
                'n_ensemble': cfg.get('n_ensemble'),
                'run_id': run.id,
                'run_name': run.name,
                'run_state': run.state,
                'n_rows_completed': run.summary.get('n_rows_completed'),
                'mean_classification_metric': run.summary.get('mean_classification_metric'),
                'mean_r2': run.summary.get('mean_r2'),
            }
            for row in per_dataset_rows:
                col_label = f"{row['dataset']}/{row['target']}" if row.get('target') else row['dataset']
                record[col_label] = row.get('metric_value')
            records.append(record)
    return records


def score(record):
    vals = [v for v in (record.get('mean_classification_metric'), record.get('mean_r2')) if v is not None]
    return sum(vals) / len(vals) if vals else None


def render_and_save(records, label, out_dir):
    import pandas as pd
    rows = [r for r in records if r['checkpoint_label'] == label]
    if not rows:
        print(f"\nNo completed runs yet for checkpoint_label={label!r}.")
        return None
    for r in rows:
        r['score'] = score(r)
    rows.sort(key=lambda r: (r['score'] is None, -(r['score'] or 0)))

    front = ['combo', 'score', 'mean_classification_metric', 'mean_r2', 'n_rows_completed',
             'dim_reduction', 'n_components', 'smoothing_steps', 'n_ensemble', 'run_name']
    df = pd.DataFrame(rows)
    other_cols = [c for c in df.columns if c not in front and c not in
                  ('checkpoint_label', 'checkpoint_path', 'run_id', 'run_state')]
    df = df[[c for c in front if c in df.columns] + sorted(other_cols)]

    csv_path = os.path.join(out_dir, f'{label}_best.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n=== {label}: {len(df)} combo(s), best first === (saved to {csv_path})")
    with pd.option_context('display.max_columns', None, 'display.width', 200, 'display.float_format', '{:.4f}'.format):
        print(df.to_string(index=False))
    return df


def render_comparison_and_save(records, baseline_label, geo_label, out_dir):
    """One row per combo with both checkpoints' scores side by side, sorted so that
    combos where geo beats baseline come first, best-baseline-first within that group."""
    import pandas as pd

    meta_cols = {'checkpoint_label', 'checkpoint_path', 'combo', 'dim_reduction', 'n_components',
                 'smoothing_steps', 'n_ensemble', 'run_id', 'run_name', 'run_state',
                 'n_rows_completed', 'mean_classification_metric', 'mean_r2', 'score'}
    dataset_cols = sorted({k for r in records for k in r if k not in meta_cols})

    by_combo = {}
    for r in records:
        if r['checkpoint_label'] in (baseline_label, geo_label):
            by_combo.setdefault(r['combo'], {})[r['checkpoint_label']] = r

    rows = []
    for combo, d in by_combo.items():
        base, geo = d.get(baseline_label), d.get(geo_label)
        if base is None and geo is None:
            continue
        base_score, geo_score = score(base) if base else None, score(geo) if geo else None
        both = base_score is not None and geo_score is not None
        template = base or geo
        row = {
            'combo': combo,
            'dim_reduction': template.get('dim_reduction'),
            'n_components': template.get('n_components'),
            'smoothing_steps': template.get('smoothing_steps'),
            'n_ensemble': template.get('n_ensemble'),
            f'{baseline_label}_score': base_score,
            f'{geo_label}_score': geo_score,
            'geo_minus_baseline': (geo_score - base_score) if both else None,
            'geo_beats_baseline': (geo_score > base_score) if both else None,
        }
        for dc in dataset_cols:
            row[f'{dc}__{baseline_label}'] = base.get(dc) if base else None
            row[f'{dc}__{geo_label}'] = geo.get(dc) if geo else None
        rows.append(row)

    if not rows:
        print(f"\nNo combos with data for either {baseline_label!r} or {geo_label!r} yet.")
        return None

    base_key, geo_key = f'{baseline_label}_score', f'{geo_label}_score'
    # geo-beats-baseline combos first; within each group, best baseline score first;
    # combos missing one side sort last within their group by whatever score exists
    def sort_key(r):
        pending = r['geo_beats_baseline'] is None
        wins = r['geo_beats_baseline'] is True
        rank_score = r[base_key] if r[base_key] is not None else (r[geo_key] or 0)
        return (pending, not wins, -(rank_score or 0))

    rows.sort(key=sort_key)
    n_both = sum(1 for r in rows if r['geo_beats_baseline'] is not None)
    n_wins = sum(1 for r in rows if r['geo_beats_baseline'] is True)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, 'comparison_best.csv')
    df.to_csv(csv_path, index=False)

    print(f"\n=== comparison: {len(rows)} combo(s) ({n_both} with both checkpoints completed, "
          f"geo beats baseline on {n_wins}/{n_both}) === (saved to {csv_path})")
    front = ['combo', base_key, geo_key, 'geo_minus_baseline', 'geo_beats_baseline',
             'dim_reduction', 'n_components', 'smoothing_steps', 'n_ensemble']
    with pd.option_context('display.max_columns', None, 'display.width', 200, 'display.float_format', '{:.4f}'.format):
        print(df[front].to_string(index=False))
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--entity', default=None)
    parser.add_argument('--project', default='NodePFN-sweep-short')
    parser.add_argument('--checkpoint_labels', nargs='+', default=[
        'full',
        'geo',
        'geo_baseline_8_gpus_uniform_geo_prior', 
        'geo_baseline_less_features_8_gpus', 
        'geo_baseline_only_zscore_norm_8_gpus'
    ])
    parser.add_argument('--baseline_label', default='full')
    parser.add_argument('--geo_label', default='geo')
    parser.add_argument('--out_dir', default=os.path.join(REPO_ROOT, 'sweep_results'))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    records = collect_runs(args.entity, args.project)

    import pandas as pd
    with open(os.path.join(args.out_dir, 'all_runs_raw.json'), 'w') as f:
        json.dump(records, f, indent=2)

    render_and_save(records, args.baseline_label, args.out_dir)
    render_and_save(records, args.geo_label, args.out_dir)
    render_comparison_and_save(records, args.baseline_label, args.geo_label, args.out_dir)

    other_labels = {r['checkpoint_label'] for r in records} - {args.baseline_label, args.geo_label}
    if other_labels:
        print(f"\nNote: also saw checkpoint_label(s) {sorted(other_labels)} not requested "
              f"(pass --baseline_label/--geo_label to include them).")


if __name__ == '__main__':
    main()
