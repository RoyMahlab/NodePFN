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
import collections
import importlib.util
import os
import shlex
import subprocess
import sys
import time
import traceback

import wandb

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
NODEPFN_DIR = os.path.join(REPO_ROOT, 'nodepfn')


def _load_send_email():
    """Load utils/send_email.py by explicit path.

    A plain ``import utils`` is ambiguous here: we add nodepfn/ (which contains
    its own utils.py) to PYTHONPATH for the subprocesses, so resolve the file
    directly instead of relying on import order.
    """
    path = os.path.join(REPO_ROOT, 'utils', 'send_email.py')
    spec = importlib.util.spec_from_file_location('nodepfn_send_email', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.send_email


def notify(subject, body):
    """Best-effort email; never let a mail failure mask the real outcome."""
    try:
        send_email = _load_send_email()
        send_email(subject, body)
    except Exception as exc:  # noqa: BLE001 - notification must not crash the run
        print(f"WARNING: failed to send notification email: {exc!r}")


class StageError(Exception):
    """A pipeline stage exited non-zero; carries the captured output tail."""

    def __init__(self, stage, cmd, returncode, tail):
        self.stage = stage
        self.cmd = cmd
        self.returncode = returncode
        self.tail = tail
        super().__init__(f"stage '{stage}' failed with exit code {returncode}")


def run(cmd, env, stage='command'):
    """Run a subprocess, streaming its output live while keeping a tail buffer.

    On non-zero exit raises StageError carrying the last lines of output so the
    caller can include *why* it failed in the notification email.
    """
    print(f"\n>>> {' '.join(shlex.quote(c) for c in cmd)}")
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    tail = collections.deque(maxlen=80)
    pending = ''
    fd = proc.stdout.fileno()
    while True:
        data = os.read(fd, 65536)  # returns as soon as any output is available
        if not data:
            break
        text = data.decode('utf-8', errors='replace')
        sys.stdout.write(text)
        sys.stdout.flush()
        pending += text
        while '\n' in pending:
            line, pending = pending.split('\n', 1)
            clean = line.rsplit('\r', 1)[-1].strip()  # keep final render of tqdm lines
            if clean:
                tail.append(clean[:500])
    proc.wait()
    if pending.strip():
        tail.append(pending.rsplit('\r', 1)[-1].strip()[:500])
    if proc.returncode != 0:
        raise StageError(stage, cmd, proc.returncode, list(tail))


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
    parser.add_argument('--max_num_classes', type=int, default=100,
                        help='classification head width / max classes the prior may sample '
                             '(default: 100; set to support high-cardinality datasets)')
    parser.add_argument('--compat_mode', type=str, default='subset',
                        choices=['subset', 'exact', 'stratify'],
                        help="context/eval class-split policy passed to pretrain (default: subset)")
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

    run_label = f"{run_name} (run id {run_id}, prior={args.prior}, gpus={args.gpus})"
    started_at = time.strftime('%Y-%m-%d %H:%M:%S')
    t0 = time.time()

    def _elapsed():
        secs = int(time.time() - t0)
        h, rem = divmod(secs, 3600)
        m, s = divmod(rem, 60)
        return f"{h}h{m:02d}m{s:02d}s"

    try:
        _run_pipeline(args, run_id, run_name, base_model_path)
    except StageError as exc:
        tail = '\n'.join(exc.tail) or '(no output captured)'
        body = (f"Training FAILED.\n\n"
                f"Run:        {run_label}\n"
                f"Stage:      {exc.stage}\n"
                f"Exit code:  {exc.returncode}\n"
                f"Command:    {' '.join(shlex.quote(c) for c in exc.cmd)}\n"
                f"Started:    {started_at}\n"
                f"Elapsed:    {_elapsed()}\n\n"
                f"--- last output ---\n{tail}\n")
        notify(f"[NodePFN] FAILED: {run_name} ({exc.stage})", body)
        raise SystemExit(f"command failed with exit code {exc.returncode}: {exc.cmd[0]}")
    except KeyboardInterrupt:
        body = (f"Training INTERRUPTED (KeyboardInterrupt / SIGINT).\n\n"
                f"Run:      {run_label}\n"
                f"Started:  {started_at}\n"
                f"Elapsed:  {_elapsed()}\n")
        notify(f"[NodePFN] INTERRUPTED: {run_name}", body)
        raise
    except BaseException as exc:  # noqa: BLE001 - report any orchestration failure
        body = (f"Training FAILED (orchestrator error, not a training subprocess).\n\n"
                f"Run:      {run_label}\n"
                f"Error:    {type(exc).__name__}: {exc}\n"
                f"Started:  {started_at}\n"
                f"Elapsed:  {_elapsed()}\n\n"
                f"--- traceback ---\n{traceback.format_exc()}\n")
        notify(f"[NodePFN] FAILED: {run_name} (orchestrator)", body)
        raise
    else:
        body = (f"Training SUCCEEDED.\n\n"
                f"Run:      {run_label}\n"
                f"Started:  {started_at}\n"
                f"Elapsed:  {_elapsed()}\n\n"
                f"Pretraining + baselines logged to one wandb run: {run_id}\n")
        notify(f"[NodePFN] SUCCESS: {run_name}", body)


def _run_pipeline(args, run_id, run_name, base_model_path):
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
                          ('--aggregate_k_gradients', args.aggregate_k_gradients),
                          ('--max_num_classes', args.max_num_classes),
                          ('--compat_mode', args.compat_mode)):
            if val is not None:
                pre_cmd += [flag, str(val)]
        pre_env = base_env()
        pre_env['WANDB_RUN_ID'] = run_id
        run(pre_cmd, pre_env, stage='pretrain')

    # --- stage 2: baselines, resuming the same run ---
    base_cmd = [sys.executable, 'log_baseline_to_wandb.py',
                '--wandb_run_id', run_id,
                '--wandb_project', args.wandb_project,
                '--base_model_path', base_model_path]
    if args.wandb_entity:
        base_cmd += ['--wandb_entity', args.wandb_entity]
    if args.datasets:
        base_cmd += ['--datasets', *args.datasets]
    run(base_cmd, base_env(), stage='baselines')

    print(f"\nDone. Pretraining + baselines logged to one wandb run: {run_id}")


if __name__ == '__main__':
    main()
