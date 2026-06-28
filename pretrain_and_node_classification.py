#!/usr/bin/env python
"""Pretrain a NodePFN model, then evaluate the node-classification baselines
and log everything to the SAME wandb run.

Flow:
  1. Generate a wandb run id up front.
  2. Run ``nodepfn/pretrain.py --model_name <name> --wandb`` with that id pinned
     via the WANDB_RUN_ID env var (wandb honours it because pretrain/train.py
     does not pass an explicit id). Checkpoints land in models_ckpts/<name>/.
  3. Run ``log_baseline_to_wandb.py --wandb_run_id <id>`` which resumes that same
     run and appends the baseline summary table + per-dataset metrics. By default
     the baselines evaluate the model we just pretrained.

Usage (from repo root):
    python pretrain_and_baseline.py --model_name my_run
    python pretrain_and_baseline.py --model_name my_run --datasets cora citeseer
"""
import argparse
import os
import shlex
import subprocess
import sys

import wandb

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
NODEPFN_DIR = os.path.join(REPO_ROOT, 'nodepfn')


def base_env():
    """Env with nodepfn/ on PYTHONPATH (needed by the package's flat imports)."""
    env = dict(os.environ)
    env['PYTHONPATH'] = NODEPFN_DIR + os.pathsep + env.get('PYTHONPATH', '')
    return env


def run(cmd, env):
    print(f"\n>>> {' '.join(shlex.quote(c) for c in cmd)}")
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    if proc.returncode != 0:
        raise SystemExit(f"command failed with exit code {proc.returncode}: {cmd[0]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model_name', required=True,
                        help='pretrain model name (checkpoints -> models_ckpts/<name>/)')
    parser.add_argument('--prior', type=str, default='geo', choices=['geo', 'causal'],
                        help="graph prior to pretrain on: 'geo' = casual_graph_generation "
                             "similarity prior, 'causal' = original MLP + SBM/random prior bag")
    parser.add_argument('--geo_similarity', type=str, default=None,
                        choices=['cosine', 'bilinear', 'mlp'],
                        help='pin the geo prior similarity kernel (default: sample it per graph)')
    parser.add_argument('--wandb_project', default='NodePFN')
    parser.add_argument('--wandb_entity', default=None)
    parser.add_argument('--wandb_run_name', default=None,
                        help='wandb run name (default: model_name)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for reproducible pretraining (default: 42)')
    parser.add_argument('--resume_epoch', type=int, default=None,
                        help='resume pretraining from this epoch checkpoint')
    parser.add_argument('--gpus', type=int, default=2,
                        help='number of GPUs for pretraining; >1 launches via torchrun (DDP)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='override pretrain epochs (work-reduction lever)')
    parser.add_argument('--num_steps', type=int, default=None,
                        help='override optimizer steps per epoch (work-reduction lever)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='override pretrain batch size')
    parser.add_argument('--aggregate_k_gradients', type=int, default=None,
                        help='override gradient accumulation (1 = full real batch per step)')
    parser.add_argument('--base_model_path', default=None,
                        help='model dir the baselines evaluate '
                             '(default: models_ckpts/<model_name>, i.e. the model just trained)')
    parser.add_argument('--datasets', nargs='*', default=None,
                        help='only evaluate these datasets (default: all in run_baseline.sh)')
    parser.add_argument('--skip_pretrain', action='store_true',
                        help='skip pretraining; requires --wandb_run_id to join an existing run')
    parser.add_argument('--wandb_run_id', default=None,
                        help='use this run id instead of generating one '
                             '(required with --skip_pretrain)')
    args = parser.parse_args()

    run_name = args.wandb_run_name or args.model_name
    base_model_path = args.base_model_path or os.path.join('models_ckpts', args.model_name)

    if args.skip_pretrain:
        if not args.wandb_run_id:
            raise SystemExit("--skip_pretrain requires --wandb_run_id")
        run_id = args.wandb_run_id
    else:
        run_id = args.wandb_run_id or wandb.util.generate_id()

    print(f"wandb run id: {run_id}  (project={args.wandb_project}, name={run_name})")

    # --- stage 1: pretrain, pinning the wandb run id via env ---
    if not args.skip_pretrain:
        # launcher: single process, or torchrun across N GPUs (DDP)
        if args.gpus > 1:
            launcher = [sys.executable, '-m', 'torch.distributed.run',
                        '--nproc_per_node', str(args.gpus), '--master_port', '29501']
        else:
            launcher = [sys.executable]
        pre_cmd = launcher + ['-m', 'nodepfn.pretrain',
                              '--model_name', args.model_name,
                              '--prior', args.prior,
                              '--seed', str(args.seed),
                              '--wandb',
                              '--wandb_project', args.wandb_project,
                              '--wandb_run_name', run_name]
        if args.geo_similarity is not None:
            pre_cmd += ['--geo_similarity', args.geo_similarity]
        if args.wandb_entity:
            pre_cmd += ['--wandb_entity', args.wandb_entity]
        if args.resume_epoch is not None:
            pre_cmd += ['--resume_epoch', str(args.resume_epoch)]
        for flag, val in (('--epochs', args.epochs), ('--num_steps', args.num_steps),
                          ('--batch_size', args.batch_size),
                          ('--aggregate_k_gradients', args.aggregate_k_gradients)):
            if val is not None:
                pre_cmd += [flag, str(val)]
        pre_env = base_env()
        pre_env['WANDB_RUN_ID'] = run_id
        run(pre_cmd, pre_env)

    # --- stage 2: baselines, resuming the same run ---
    base_cmd = [sys.executable, 'log_baseline_to_wandb.py',
                '--wandb_run_id', run_id,
                '--wandb_project', args.wandb_project,
                '--base_model_path', base_model_path]
    if args.wandb_entity:
        base_cmd += ['--wandb_entity', args.wandb_entity]
    if args.datasets:
        base_cmd += ['--datasets', *args.datasets]
    run(base_cmd, base_env())

    print(f"\nDone. Pretraining + baselines logged to one wandb run: {run_id}")


if __name__ == '__main__':
    main()
