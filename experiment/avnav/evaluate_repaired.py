"""Evaluate a declared candidate, or the validation-selected frozen final policy."""
import argparse
import json
from pathlib import Path
import torch

from experiment.avnav.benchmark_runtime import BENCHMARK_ROOT, DATA_ROOT
from experiment.avnav.evaluate_codex_saccql_lstm import (
    _evaluate_split, _sha256_file, _prepare_runtime_dataset_manifest, _atomic_json,
)


def verify_frozen(run_dir, frozen):
    for path, expected in frozen['immutable_files'].items():
        if _sha256_file(path) != expected:
            raise RuntimeError(f'Frozen file changed: {path}')
    for record in frozen['validation_results']:
        path = Path(record['path'])
        if _sha256_file(path) != record['sha256']:
            raise RuntimeError('Validation artifact changed after selection')
        result = json.loads(path.read_text())
        if result['status'] != 'pass' or result['partial_validation_smoke']:
            raise RuntimeError('Final test requires complete error-free validation')
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--split', required=True, choices=['val_multiple', 'val_multiple_unheard', 'test_multiple', 'test_multiple_unheard'])
    parser.add_argument('--checkpoint')
    parser.add_argument('--training-config')
    parser.add_argument('--output-dir')
    parser.add_argument('--episode-limit', type=int)
    parser.add_argument('--episode-start', type=int, default=0)
    parser.add_argument('--episode-count', type=int)
    parser.add_argument('--input-ablation', choices=['none', 'vision_only', 'audio_only'], default='none')
    parser.add_argument('--policy-modality', choices=['multimodal', 'vision_only', 'audio_only'], default='multimodal')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.input_ablation != 'none' and not args.split.startswith('val'):
        raise ValueError('Modality ablations are restricted to validation splits')
    if args.policy_modality != 'multimodal' and not args.split.startswith('val'):
        raise ValueError('Trained single-modality policies are restricted to validation splits')
    if args.policy_modality != 'multimodal' and args.input_ablation != 'none':
        raise ValueError('Choose a trained modality policy or an input ablation, not both')
    if (args.episode_start or args.episode_count is not None) and not args.split.startswith('val'):
        raise ValueError('Episode sharding is restricted to validation splits')
    run_dir = Path(args.run_dir).resolve()
    protocol = json.loads((run_dir / 'run_protocol.json').read_text())
    torch.set_num_threads(2)
    config = BENCHMARK_ROOT / 'configs/audiogoal.yaml'
    if args.split.startswith('test'):
        if any([args.checkpoint, args.training_config, args.output_dir, args.episode_limit]):
            raise ValueError('Test accepts only frozen selection; no overrides')
        frozen = verify_frozen(run_dir, json.loads((run_dir / 'frozen_selection.json').read_text()))
        audit = json.loads((run_dir / 'audit/data_source_audit.json').read_text())
        declared = audit['validation_and_test_split_manifests'][args.split]
        if _sha256_file(declared['path']) != declared['sha256']:
            raise RuntimeError('Test dataset manifest changed since the pre-training audit')
        checkpoint = Path(frozen['checkpoint'])
        training_config = Path(frozen['training_config'])
        output_dir = run_dir / 'evaluation/test'
        artifact = _prepare_runtime_dataset_manifest(args.split, DATA_ROOT, run_dir, source_manifest=declared['path'])
        manifest = artifact['runtime_manifest']
        _atomic_json(run_dir / f'evaluation/{args.split}_manifest.json', artifact)
    else:
        if not args.checkpoint or not args.training_config or not args.output_dir:
            raise ValueError('Validation requires explicit checkpoint, configuration, and output directory')
        checkpoint = Path(args.checkpoint).resolve()
        training_config = Path(args.training_config).resolve()
        output_dir = Path(args.output_dir).resolve()
        manifest = None
    result_path = output_dir / f'{args.split}.json'
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if previous['status'] == 'pass':
            raise FileExistsError(f'Completed evaluation already exists: {result_path}')
    result = _evaluate_split(
        args.split, run_dir / f'audit/split_episode_ids/{args.split}.jsonl',
        run_dir, torch.device(args.device), protocol['evaluation']['seed'],
        protocol['evaluation']['max_steps'], output_dir, config, DATA_ROOT,
        dataset_manifest_override=manifest, checkpoint=checkpoint, training_config=training_config,
        episode_limit=args.episode_limit, input_ablation=args.input_ablation,
        episode_start=args.episode_start, episode_count=args.episode_count,
        policy_modality=args.policy_modality,
    )
    if result['status'] != 'pass':
        raise RuntimeError(f'Evaluation incomplete: {result_path}')


if __name__ == '__main__':
    main()
