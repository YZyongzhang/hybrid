"""Choose a checkpoint by validation rollout SPL, then evaluate the held-out test split."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time

from experiment.avwan.data_pipeline.prepare import write_json
from experiment.avwan.evaluate import aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--foundation-checkpoint', help='Evaluate recurrent SAC candidates with this frozen encoder')
    parser.add_argument('--seed', type=int, default=20260925)
    parser.add_argument('--validation-source', default='/data/Getuanhui/avwan_telephone_greedy_val')
    parser.add_argument('--validation-split', default='val_telephone')
    parser.add_argument('--test-split', default='test_telephone')
    parser.add_argument('--config', help='Experiment environment configuration passed to each rollout')
    parser.add_argument('--test-max-episodes', type=int, default=0)
    parser.add_argument('--test-workers', type=int, default=4)
    parser.add_argument('--test-concurrency', type=int, default=2,
                        help='Maximum simultaneous test processes; large acoustic scenes use substantial RAM')
    parser.add_argument('--validation-concurrency', type=int, default=1)
    parser.add_argument('--process-episodes', type=int, default=25,
                        help='Restart simulator processes between episode batches to release native allocations')
    parser.add_argument('--process-rss-mb', type=int, default=8192)
    parser.add_argument('--wait-validation', action='store_true', help='Wait for already-running validation jobs')
    args = parser.parse_args()
    run = Path(args.run).resolve()
    output = run / 'rollouts'
    output.mkdir(exist_ok=True)
    candidates = json.loads((run / 'top_checkpoints.json').read_text())
    if args.foundation_checkpoint:
        # Different offline criteria can select the same epoch; evaluate it once.
        candidates = list({item['epoch']: item for item in candidates}.values())
    if args.wait_validation:
        deadline = time.monotonic() + 7200
        while True:
            pending = []
            for item in candidates:
                path = output / f'val_epoch_{item["epoch"]:03d}' / 'summary.json'
                summary = json.loads(path.read_text()) if path.exists() else {}
                if summary.get('episodes', -1) != summary.get('planned', 0):
                    pending.append(item['epoch'])
            if not pending:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f'Validation jobs did not finish: {pending}')
            time.sleep(5)
    comparison = []
    def evaluate(checkpoint, folder, split, selection=None, limit=0, shard=0, shards=1):
        command = [sys.executable, '-m', 'experiment.avwan.evaluate', '--checkpoint', str(checkpoint),
                   '--output', str(folder), '--split', split, '--resume']
        command += ['--seed', str(args.seed)]
        if args.config:
            command += ['--config', str(Path(args.config).resolve())]
        if args.foundation_checkpoint:
            command += ['--foundation-checkpoint', str(Path(args.foundation_checkpoint).resolve())]
        command += ['--shards', str(shards), '--shard-index', str(shard)]
        command += ['--max-new-episodes', str(args.process_episodes), '--max-rss-mb', str(args.process_rss_mb)]
        if selection:
            command += ['--selection-source', selection]
        if limit:
            command += ['--max-episodes', str(limit)]
        print('Evaluating ' + checkpoint.name + ' on ' + split, flush=True)
        summary_path = folder / 'summary.json'
        previous = json.loads(summary_path.read_text()).get('episodes', 0) if summary_path.exists() else 0
        while True:
            with folder.with_suffix('.log').open('a') as log:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
            summary = json.loads(summary_path.read_text())
            if summary['errors']:
                raise RuntimeError(f'Erroneous rollout: {folder}')
            if summary['episodes'] == summary['planned']:
                break
            if summary['episodes'] <= previous:
                raise RuntimeError(f'No evaluation progress: {folder}')
            previous = summary['episodes']
            print(f'{folder.name}: saved {previous}/{summary["planned"]}; restarting simulator process', flush=True)
        print(json.dumps({k: summary[k] for k in ('epoch','episodes','spl','success_rate')}), flush=True)
        return summary
    def validate_candidate(item):
        summary = evaluate(run / item['file'], output / f'val_epoch_{item["epoch"]:03d}',
                           args.validation_split, args.validation_source)
        return dict(**item, rollout=summary)
    if args.validation_concurrency < 1:
        raise ValueError('validation-concurrency must be positive')
    with ThreadPoolExecutor(max_workers=args.validation_concurrency) as pool:
        futures = [pool.submit(validate_candidate, item) for item in candidates]
        for future in as_completed(futures):
            comparison.append(future.result())
            comparison.sort(key=lambda item: item['epoch'])
            write_json(output / 'validation_comparison.json', comparison)
    # Older runs include offline validation loss as a tie-breaker. Train-only
    # runs expose just the final epoch, so navigation metrics select it directly.
    best = max(comparison, key=lambda x: (x['rollout']['spl'], x['rollout']['success_rate'],
                                         -x.get('val_loss', 0.)))
    write_json(output / 'selected_checkpoint.json', best)
    import shutil
    shutil.copyfile(run / best['file'], run / 'best_spl.pt')
    if args.test_workers < 1 or args.test_concurrency < 1:
        raise ValueError('test-workers and test-concurrency must be positive')
    if args.test_workers == 1:
        test = evaluate(run / best['file'], output / 'test_selected', args.test_split, limit=args.test_max_episodes)
    else:
        folder = output / 'test_selected'
        folder.mkdir(exist_ok=True)
        def worker(index):
            return evaluate(run / best['file'], folder / f'worker_{index}', args.test_split,
                            limit=args.test_max_episodes, shard=index, shards=args.test_workers)
        with ThreadPoolExecutor(max_workers=min(args.test_workers, args.test_concurrency)) as pool:
            summaries = list(pool.map(worker, range(args.test_workers)))
        for summary in summaries[1:]:
            for key in ('checkpoint_sha256', 'foundation_sha256', 'split', 'seed', 'policy'):
                if summary.get(key) != summaries[0].get(key):
                    raise RuntimeError(f'Evaluation workers disagree on {key}')
        rows, selections = [], []
        for index in range(args.test_workers):
            child = folder / f'worker_{index}'
            rows.extend(json.loads(line) for line in (child / 'episodes.jsonl').read_text().splitlines())
            selections.extend(json.loads((child / 'selection.json').read_text()))
        keys = {(r['scene'], r['episode_id']) for r in rows}
        if len(keys) != len(rows) or keys != {tuple(x) for x in selections}:
            raise RuntimeError('Duplicate/missing test episodes across workers')
        metadata = json.loads((folder / 'worker_0/run.json').read_text())
        metadata.pop('shards', None)
        metadata.pop('shard_index', None)
        import hashlib
        metadata.update(planned=len(rows), workers=args.test_workers,
                        selection_sha256=hashlib.sha256(json.dumps(sorted(selections)).encode()).hexdigest())
        test = aggregate(rows, metadata)
        write_json(folder / 'summary.json', test)
        write_json(folder / 'selection.json', sorted(selections))
        (folder / 'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in sorted(rows, key=lambda r:(r['scene'],r['episode_id']))))
    if all(item.get('criterion') == 'epoch_snapshot' for item in candidates):
        criterion = 'navigation validation SPL among saved training epochs'
    elif all(item.get('criterion') == 'final_epoch' for item in candidates):
        criterion = 'navigation validation SPL for the final training epoch'
    else:
        criterion = ('validation rollout SPL among best critic loss, best policy NLL, and final epoch'
                     if args.foundation_checkpoint else
                     'validation rollout SPL among top offline-loss checkpoints')
    if args.test_max_episodes and test['episodes'] != args.test_max_episodes:
        raise RuntimeError('Test episode count differs from the requested random sample size')
    write_json(output / 'final_results.json', dict(selection=criterion,
               validation=comparison, selected=best['file'], test=test))
    print('Finished validation selection and held-out testing', flush=True)


if __name__ == '__main__':
    main()
