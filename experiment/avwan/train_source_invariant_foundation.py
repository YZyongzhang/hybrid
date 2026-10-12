"""Fine-tune AVWAN with source-robust binaural features and paired-view loss.

Both views keep the same physical left/right direction. The second view changes
shared source coloration and removes a short frequency band, so the policy must
predict the same waypoint without relying on sound identity. Only train scenes
are used for gradients; heard/unheard validation demonstrations select an epoch.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .data_pipeline.dataset import make_loader
from .data_pipeline.prepare import write_json
from .model.hybrid import Network
from .model.maps import unpack_geometry
from .train_unheard import augment_acoustics
from .training.trainer import save_checkpoint


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / 'result/2026-10-11-large-foundation-study/composite_cache'
DEFAULT_INITIAL = ROOT / 'result/2026-10-11-large-foundation-study/ratio_big_8ep/training/best.pt'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source_view(batch, strength, dropout_probability):
    """Perturb the shared source spectrum without changing interaural sign."""
    view = augment_acoustics(batch, strength=strength)
    magnitude = torch.expm1(view['spectrogram'].float().clamp(0, 20))
    count, frequencies = magnitude.shape[:2]
    device = magnitude.device
    center = torch.randint(0, frequencies, (count, 1), device=device)
    width = torch.randint(3, 9, (count, 1), device=device)
    distance = (torch.arange(frequencies, device=device)[None] - center).abs()
    remove = ((distance <= width) &
              (torch.rand(count, 1, device=device) < dropout_probability))
    magnitude = magnitude.masked_fill(remove[:, :, None, None], 0.)
    view['spectrogram'] = torch.log1p(magnitude)
    return view


def source_profile_view(batch, strength=.7):
    """Mask bands weak in a different real sound while preserving direction.

    A donor supplies frequency availability, averaged over time and both ears.
    Remaining binaural ratios and all navigation targets stay unchanged.
    """
    if 'sound_label' not in batch:
        raise ValueError('Source-style swap requires aligned sound labels')
    labels = batch['sound_label']
    if len(labels) < 2 or labels.unique().numel() < 2:
        raise ValueError('Source-style swap needs at least two sound classes per batch')
    view = dict(batch)
    magnitude = torch.expm1(batch['spectrogram'].float().clamp(0, 20))
    # Mean over time and both ears removes interaural direction from the donor.
    profile = magnitude.mean(dim=(2, 3))
    donor = torch.randperm(len(labels), device=labels.device)
    collisions = labels[donor] == labels
    if collisions.any():
        unequal = labels[:, None] != labels[None, :]
        replacement = unequal.float().argmax(dim=1)
        donor = torch.where(collisions, replacement, donor)
    bands = round(profile.shape[1] * strength * .5)
    mask = torch.zeros_like(profile, dtype=torch.bool)
    if bands:
        low_bands = profile[donor].argsort(dim=1)[:, :bands]
        mask.scatter_(1, low_bands, True)
    view['spectrogram'] = torch.log1p(magnitude.masked_fill(mask[:, :, None, None], 0.))
    return view


class ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, scale):
        ctx.scale = scale
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.scale * gradient, None


def predict(model, batch, return_audio=False):
    rgb = torch.cat((batch['prev_rgb'], batch['rgb']), dim=-1).float() / 255.
    depth = torch.cat((batch['prev_depth'], batch['depth']), dim=-1).float()
    return model.forward_joint_features(batch['spectrogram'], rgb, depth,
                                        unpack_geometry(batch['gm']), batch['am'],
                                        batch['action_mask'], return_audio=return_audio)


def supervised(logits, angles, batch, action_count):
    action_loss = F.cross_entropy(logits, batch['action'])
    valid = batch['action'] != action_count // 2
    angle_loss = (F.cross_entropy(angles[valid], batch['angle_label'][valid])
                  if valid.any() else angles.sum() * 0.)
    accuracy = (logits.argmax(-1) == batch['action']).float().mean()
    return action_loss, angle_loss, accuracy


def paired_objective(model, batch, config, sound_classifier=None):
    swap = (config.get('augmentation_type') == 'source_profile' and
            batch['sound_label'].unique().numel() > 1)
    perturbed = (source_profile_view(batch, config['source_profile_strength'])
                 if swap else
                 source_view(batch, config['augmentation_strength'],
                             config['frequency_dropout_probability']))
    if sound_classifier is None:
        logits_a, angles_a, features_a = predict(model, batch)
    else:
        logits_a, angles_a, features_a, audio_a = predict(model, batch, return_audio=True)
    logits_b, angles_b, features_b = predict(model, perturbed)
    action_a, angle_a, accuracy = supervised(logits_a, angles_a, batch, model.action_count)
    action_b, angle_b, _ = supervised(logits_b, angles_b, batch, model.action_count)
    action_loss = (action_a + action_b) * .5
    angle_loss = (angle_a + angle_b) * .5
    # Each view is a detached target for the other. Invalid actions retain
    # zero probability because forward_joint_features applies the same mask.
    log_a = F.log_softmax(logits_a.float(), dim=-1)
    log_b = F.log_softmax(logits_b.float(), dim=-1)
    consistency = .5 * (F.kl_div(log_a, log_b.detach().exp(), reduction='batchmean') +
                        F.kl_div(log_b, log_a.detach().exp(), reduction='batchmean'))
    representation = (1. - F.cosine_similarity(features_a.float(), features_b.float(), dim=-1)).mean()
    loss = (action_loss + config['angle_loss_weight'] * angle_loss +
            config['action_consistency_weight'] * consistency +
            config['representation_consistency_weight'] * representation)
    metrics = dict(action_loss=action_loss, angle_loss=angle_loss,
                   action_accuracy=accuracy, action_consistency=consistency,
                   representation_consistency=representation)
    if sound_classifier is not None:
        sound_logits = sound_classifier(ReverseGradient.apply(
            audio_a.float(), config['sound_reversal_scale']))
        sound_loss = F.cross_entropy(sound_logits, batch['sound_label'])
        loss = loss + config['sound_adversary_weight'] * sound_loss
        metrics['sound_loss'] = sound_loss
        metrics['sound_accuracy'] = (sound_logits.argmax(-1) == batch['sound_label']).float().mean()
    return loss, metrics


def train(args):
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    output.mkdir(parents=True)
    train_dir = output / 'training'
    train_dir.mkdir()
    cache, initial = args.cache.resolve(), args.initial.resolve()
    seed = args.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    state = torch.load(initial, map_location='cpu', weights_only=False)
    network = dict(state['config']['network'])
    if network['audio_normalization'] != 'binaural_ratio':
        raise ValueError('Initialization must be the trained binaural-ratio foundation')
    network['audio_normalization'] = args.audio_normalization
    model = Network(**network).to(device)
    model.load_state_dict(state['model'], strict=True)
    model.requires_grad_(False)
    for module in (model.audio_encoder, model.vaencoder, model.final.fc1,
                   model.final.action_head, model.detail_encoder):
        if module is not None:
            module.requires_grad_(True)
    model.modality_embedding.requires_grad_(True)
    sound_classifier = None
    if args.sound_adversary_weight:
        if args.sound_labels is None:
            raise ValueError('--sound-labels is required with a sound adversary')
        label_manifest = json.loads((args.sound_labels / 'manifest.json').read_text())
        sound_classifier = nn.Sequential(nn.Linear(128, 128), nn.ReLU(),
                                         nn.Linear(128, len(label_manifest['classes']))).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if sound_classifier is not None:
        trainable += list(sound_classifier.parameters())
    data = dict(cache=str(cache))
    if sound_classifier is not None or args.augmentation_type == 'source_profile':
        if args.sound_labels is None:
            raise ValueError('--sound-labels is required with a source profile view')
        data['sound_labels'] = str(args.sound_labels.resolve())
        data['sound_labels_manifest_sha256'] = digest(args.sound_labels / 'manifest.json')
    config = dict(seed=seed, network=network, data=data,
                  initialization=dict(checkpoint=str(initial), sha256=digest(initial)),
                  training=dict(output=str(train_dir), epochs=args.epochs,
                                batch_size=args.batch_size, workers=args.workers,
                                prefetch_factor=2, pin_memory=True,
                                learning_rate=args.learning_rate, weight_decay=1e-4,
                                angle_loss_weight=.5,
                                augmentation_type=args.augmentation_type,
                                augmentation_strength=args.augmentation_strength,
                                source_profile_strength=args.source_profile_strength,
                                frequency_dropout_probability=args.frequency_dropout_probability,
                                action_consistency_weight=args.action_consistency_weight,
                                representation_consistency_weight=args.representation_consistency_weight))
    if sound_classifier is not None:
        config['training'].update(sound_adversary_weight=args.sound_adversary_weight,
                                  sound_reversal_scale=args.sound_reversal_scale)
    write_json(output / 'config.json', config)
    write_json(train_dir / 'trainable_parameters.json',
               dict(names=[name for name, parameter in model.named_parameters()
                           if parameter.requires_grad] +
                          (['sound_classifier.' + name for name, _ in sound_classifier.named_parameters()]
                           if sound_classifier is not None else []),
                    trainable=sum(p.numel() for p in trainable),
                    total=sum(p.numel() for p in model.parameters()) +
                          (sum(p.numel() for p in sound_classifier.parameters())
                           if sound_classifier is not None else 0)))
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    loaders = {split: make_loader(config, split) for split in ('train', 'val', 'val_heard')}
    signature = deepcopy(state['signature'])
    signature['architecture'] = 'avwan_hybrid_source_invariant_v1'
    write_json(output / 'status.json', dict(status='training'))

    def run(split, epoch):
        update = split == 'train'
        model.train(update)
        if sound_classifier is not None:
            sound_classifier.train(update)
        # Frozen modules must not update BatchNorm/dropout statistics.
        if update:
            for module in (model.visual_encoder, model.visual_transformer_encoder,
                           model.geometric_encoder, model.acoustic_encoder,
                           model.mask_encoder):
                module.eval()
        loaders[split].dataset.set_epoch(epoch)
        totals, seen = {}, 0
        started = time.monotonic()
        for batch in loaders[split]:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.set_grad_enabled(update), torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                if update:
                    loss, metrics = paired_objective(model, batch, config['training'],
                                                     sound_classifier)
                else:
                    logits, angles, _ = predict(model, batch)
                    action, angle, accuracy = supervised(logits, angles, batch, model.action_count)
                    loss = action + config['training']['angle_loss_weight'] * angle
                    metrics = dict(action_loss=action, angle_loss=angle, action_accuracy=accuracy)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{split}: non-finite loss')
            if update:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 10.)
                scaler.step(optimizer); scaler.update()
            n = len(batch['action']); seen += n
            for key, value in dict(loss=loss, **metrics).items():
                totals[key] = totals.get(key, 0.) + float(value.detach()) * n
        return dict(**{key: value / seen for key, value in totals.items()},
                    samples=seen, seconds=time.monotonic() - started)

    write_json(train_dir / 'initial_validation.json',
               {split: run(split, 0) for split in ('val', 'val_heard')})
    best = float('inf')
    for epoch in range(1, args.epochs + 1):
        row = dict(epoch=epoch,
                   train=run('train', epoch),
                   val=run('val', epoch),
                   val_heard=run('val_heard', epoch))
        row['worst_validation_action_loss'] = max(row['val']['action_loss'],
                                                  row['val_heard']['action_loss'])
        with (train_dir / 'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
        payload = dict(model=model.state_dict(), signature=signature, config=config,
                       epoch=epoch, validation=row,
                       selection='minimum worse heard/unheard validation action NLL')
        if sound_classifier is not None:
            payload['sound_classifier'] = sound_classifier.state_dict()
        save_checkpoint(train_dir / 'latest.pt', payload)
        if row['worst_validation_action_loss'] < best:
            best = row['worst_validation_action_loss']
            save_checkpoint(train_dir / 'best.pt', payload)
            write_json(train_dir / 'best.json', dict(epoch=epoch,
                       worst_validation_action_loss=best,
                       sha256=digest(train_dir / 'best.pt')))
    write_json(output / 'status.json', dict(status='trained', best_validation_action_loss=best))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--initial', type=Path, default=DEFAULT_INITIAL)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=3e-5)
    parser.add_argument('--augmentation-strength', type=float, default=1.5)
    parser.add_argument('--augmentation-type', choices=('synthetic', 'source_profile'),
                        default='synthetic')
    parser.add_argument('--source-profile-strength', type=float, default=.7)
    parser.add_argument('--audio-normalization', choices=('binaural_ratio', 'binaural_robust_ratio'),
                        default='binaural_robust_ratio')
    parser.add_argument('--frequency-dropout-probability', type=float, default=.5)
    parser.add_argument('--action-consistency-weight', type=float, default=.5)
    parser.add_argument('--representation-consistency-weight', type=float, default=.05)
    parser.add_argument('--sound-labels', type=Path)
    parser.add_argument('--sound-adversary-weight', type=float, default=0.)
    parser.add_argument('--sound-reversal-scale', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=20261012)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if (args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or
            args.learning_rate <= 0 or args.augmentation_strength <= 0 or
            not 0 <= args.source_profile_strength <= 1 or
            not 0 <= args.frequency_dropout_probability <= 1 or
            args.action_consistency_weight < 0 or args.representation_consistency_weight < 0 or
            args.sound_adversary_weight < 0 or args.sound_reversal_scale < 0):
        parser.error('Invalid training hyperparameters')
    train(args)


if __name__ == '__main__':
    main()
