"""Fine-tune the AVWAN binaural-ratio foundation on a frozen large train set.

Keep the existing independent unheard and heard demonstrations for validation.
The training cache must come from a fixed ``snapshot_large_train`` directory.
"""
import argparse
import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = ROOT / 'result/2026-10-11-large-foundation-cache'
DEFAULT_VALIDATION = ROOT / 'result/2026-10-03-unheard-generalization-study/domain_cache'
DEFAULT_FOUNDATION = ROOT / 'result/2026-10-03-unheard-generalization-study/best_foundation.pt'
DEFAULT_OUTPUT = ROOT / 'result/2026-10-11-large-foundation-study'


def prepare(train_cache, validation_cache, output):
    new = json.loads((train_cache / 'manifest.json').read_text())
    old = json.loads((validation_cache / 'manifest.json').read_text())
    if new['identity']['purpose'] != 'avwan_foundation_spatial':
        raise ValueError('Not an AVWAN foundation training cache')
    if new['observation_schema'] != old['observation_schema'] or new['action_count'] != old['action_count']:
        raise ValueError('Training and validation observation schemas differ')
    manifest = copy.deepcopy(new)
    manifest['splits']['train'] = [dict(entry, path=str(train_cache / entry['path']))
                                   for entry in new['splits']['train']]
    for split in ('val', 'val_heard'):
        manifest['splits'][split] = [dict(entry, path=str(validation_cache / entry['path']))
                                     for entry in old['splits'][split]]
        manifest['samples'][split] = old['samples'][split]
    manifest['composite_sources'] = dict(train=str(train_cache),
                                         val_unheard=str(validation_cache),
                                         val_heard=str(validation_cache))
    manifest['independent_unheard_validation'] = old['independent_unheard_validation']
    cache = output / 'composite_cache'
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / 'manifest.json'
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise ValueError('Existing composite cache has different source manifests')
    target.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({split: manifest['samples'][split]
                      for split in ('train', 'val', 'val_heard')}, indent=2), flush=True)
    return cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('all', 'prepare', 'train'), default='all')
    parser.add_argument('--train-cache', type=Path, default=DEFAULT_TRAIN)
    parser.add_argument('--validation-cache', type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument('--foundation', type=Path, default=DEFAULT_FOUNDATION)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--name', default='ratio_big_8ep')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    args = parser.parse_args()
    train_cache, validation_cache = args.train_cache.resolve(), args.validation_cache.resolve()
    output, foundation = args.output.resolve(), args.foundation.resolve()
    if args.epochs < 1 or args.learning_rate <= 0:
        parser.error('epochs and learning rate must be positive')
    if args.stage != 'train':
        cache = prepare(train_cache, validation_cache, output)
    else:
        cache = output / 'composite_cache'
        if not (cache / 'manifest.json').exists():
            raise FileNotFoundError('Run --stage prepare before --stage train')
    if args.stage != 'prepare':
        folder = output / args.name
        if (folder / 'status.json').exists():
            state = json.loads((folder / 'status.json').read_text())
            if state['status'] == 'trained':
                print(f'Already trained: {folder}', flush=True)
                return
        from . import train_unheard
        train_unheard.OUT = output
        train_unheard.train(args.name, 'ratio', epochs=args.epochs,
                           learning_rate=args.learning_rate, checkpoint=foundation,
                           data_cache=cache, spatial_detail=True, freeze_backbone=True)


if __name__ == '__main__':
    main()
