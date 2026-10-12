"""Measure action stability under shared-ear source changes on held-out demos."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .data_pipeline.dataset import make_loader
from .model.hybrid import Network
from .train_source_invariant_foundation import predict, source_view


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def evaluate(checkpoint, cache, split, device, strength, dropout_probability, seed):
    torch.manual_seed(seed)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = Network(**state['config']['network']).to(device)
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    loader = make_loader(dict(seed=seed, data=dict(cache=str(cache)),
                              training=dict(batch_size=64, workers=0,
                                            pin_memory=False)), split)
    totals = dict(nll=0., perturbed_nll=0., symmetric_kl=0., changed_action=0.,
                  accuracy=0., perturbed_accuracy=0.)
    samples = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        changed = source_view(batch, strength, dropout_probability)
        logits, _, _ = predict(model, batch)
        augmented, _, _ = predict(model, changed)
        log_p, log_q = F.log_softmax(logits, -1), F.log_softmax(augmented, -1)
        p, q = log_p.exp(), log_q.exp()
        n = len(batch['action'])
        samples += n
        totals['nll'] += float(F.cross_entropy(logits, batch['action'], reduction='sum'))
        totals['perturbed_nll'] += float(F.cross_entropy(augmented, batch['action'], reduction='sum'))
        totals['symmetric_kl'] += float((.5 * ((p * (log_p - log_q)).sum(-1) +
                                              (q * (log_q - log_p)).sum(-1))).sum())
        totals['changed_action'] += float((logits.argmax(-1) != augmented.argmax(-1)).sum())
        totals['accuracy'] += float((logits.argmax(-1) == batch['action']).sum())
        totals['perturbed_accuracy'] += float((augmented.argmax(-1) == batch['action']).sum())
    return dict(split=split, samples=samples,
                **{key: value / samples for key, value in totals.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path(
        'result/2026-10-11-large-foundation-study/composite_cache'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=20261012)
    parser.add_argument('--strength', type=float, default=1.5)
    parser.add_argument('--frequency-dropout-probability', type=float, default=.5)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.strength <= 0 or not 0 <= args.frequency_dropout_probability <= 1:
        parser.error('Invalid source perturbation settings')
    torch.set_num_threads(4)
    report = dict(checkpoint=str(args.checkpoint.resolve()),
                  checkpoint_sha256=sha256(args.checkpoint),
                  cache=str(args.cache.resolve()), seed=args.seed,
                  strength=args.strength,
                  frequency_dropout_probability=args.frequency_dropout_probability,
                  splits={split: evaluate(args.checkpoint, args.cache, split, args.device,
                                          args.strength, args.frequency_dropout_probability,
                                          args.seed) for split in ('val', 'val_heard')})
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
