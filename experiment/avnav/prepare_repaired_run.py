"""Recheck split identities and source provenance before the repaired experiment."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

from experiment.avnav.train_codex_foundation import _atomic_json, _sha256


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def prepare(run_dir, previous):
    run_dir, previous = Path(run_dir).resolve(), Path(previous).resolve()
    if (run_dir / 'run_protocol.json').exists():
        raise FileExistsError('Use a new run directory')
    old = json.loads((previous / 'audit/data_source_audit.json').read_text())
    audit_dir = run_dir / 'audit'
    audit_dir.mkdir(parents=True, exist_ok=True)
    train_root = Path(old['source_train_split']['episode_pickle_root'])
    train_keys = {(p.parent.name, p.stem) for p in train_root.rglob('*.pkl')}
    assert len(train_keys) == old['source_train_split']['episode_count']
    train_manifest = Path(old['source_train_split']['foundation_pt_manifest'])
    assert _sha256(train_manifest) == old['source_train_split']['foundation_pt_manifest_sha256']
    train_pt = json.loads(train_manifest.read_text())
    assert Path(train_pt['source']).resolve() == train_root.resolve()
    assert train_pt['episodes'] == len(train_keys)
    assert all((train_manifest.parent / item['file']).is_file() for item in train_pt['shards'])
    selection = Path(old['selection_manifest'])
    assert _sha256(selection) == old['selection_manifest_sha256']
    plan = {row['task_id']: row for row in rows(selection)}
    selected = set()
    coverage = {}
    for category, detail in old['categories'].items():
        manifest = Path(detail['manifest'])
        assert _sha256(manifest) == detail['manifest_sha256']
        records = rows(manifest)
        assert len(records) == detail['manifest_rows']
        for row in records:
            assert row['status'] == 'saved' and row.get('error') is None
            assert Path(row['file']).is_file()
            key = (str(row['scene']), str(row['episode_id']))
            assert key in train_keys and row['task_id'] in plan
            expected = plan[row['task_id']]
            assert (str(expected['scene']), str(expected['episode_id'])) == key
            selected.add(key)
        coverage[category] = {
            'episodes': len(records),
            'transitions': sum(r['steps'] for r in records),
            'collision_steps': sum(r['collisions'] for r in records),
            'successful_episodes_with_collision': sum(bool(r['success']) and r['collisions'] > 0 for r in records),
            'recorded_recovery_events': sum(r.get('recoveries', 0) for r in records),
            'unresolved_events': sum(r.get('unresolved', 0) for r in records),
            'scenes': len({r['scene'] for r in records}),
        }
    for split, detail in old['validation_and_test_split_manifests'].items():
        # Hash raw manifest bytes and read only pre-existing identity inventories for test.
        assert _sha256(Path(detail['path'])) == detail['sha256']
        source_ids = previous / 'audit/split_episode_ids' / f'{split}.jsonl'
        identities = rows(source_ids)
        keys = {(str(r['scene']), str(r['episode_id'])) for r in identities}
        assert len(keys) == len(identities) == detail['episode_count']
        assert not train_keys & keys
        assert not {k[0] for k in train_keys} & {k[0] for k in keys}
        destination = audit_dir / 'split_episode_ids' / source_ids.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source_ids.read_bytes())
        detail['episode_id_snapshot'] = str(destination)
        if split.startswith('val'):
            source = Path(old['validation_pickle_inventory'][split]['pickle_root'])
            assert {(p.parent.name, p.stem) for p in source.rglob('*.pkl')} == keys
    parent_inventory = audit_dir / 'training_parent_episode_ids.jsonl'
    parent_inventory.write_text(''.join(json.dumps({'scene': s, 'episode_id': e}) + '\n' for s, e in sorted(train_keys)))
    old['source_train_split']['key_inventory'] = str(parent_inventory)
    old['source_train_split']['key_inventory_sha256'] = _sha256(parent_inventory)
    old['selected_unique_parent_episode_keys'] = len(selected)
    old['rechecked_for_run'] = str(run_dir)
    old['previous_audit_sha256'] = _sha256(previous / 'audit/data_source_audit.json')
    old['test_usage'] = 'manifest hashes and split identities only; no test metrics or trajectories read'
    _atomic_json(audit_dir / 'data_source_audit.json', old)
    _atomic_json(audit_dir / 'coverage_manifest.json', {
        'categories': coverage,
        'scope': 'train manifests only; per-transition recovery audit follows during feature extraction',
        'collection_protocol': json.loads((previous / 'run_protocol.json').read_text())['collection_provenance'],
    })
    print(json.dumps({'status': 'pass', 'train_parent_episodes': len(train_keys), 'coverage': coverage}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--previous-run', default='result/2026-10-27-avnav-codex-full-training')
    args = parser.parse_args()
    prepare(args.run_dir, args.previous_run)
