"""Copy selected successful AVNav collection trajectories into offline RL inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


def _read_jsonl(path: Path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def prepare(collections, audit_path, selection_path=None):
    fallback_selection = (Path(selection_path).expanduser().resolve()
                          if selection_path else None)
    audits = []
    global_success_keys = set()
    for source_arg in collections:
        source = Path(source_arg).expanduser().resolve()
        run = json.loads((source / "run.json").read_text(encoding="utf-8"))
        recorded_selection = run.get("selection")
        selected_file = (Path(recorded_selection).expanduser().resolve()
                         if recorded_selection else fallback_selection)
        if selected_file is None:
            raise ValueError(f"No episode selection recorded for {source}")
        selection = json.loads(selected_file.read_text(encoding="utf-8"))
        selection_keys = [(str(row["scene"]), str(row["episode_id"])) for row in selection]
        if len(set(selection_keys)) != len(selection_keys):
            raise ValueError(f"Selection contains duplicate scene/episode keys: {selected_file}")
        start, end = (int(value) for value in run["episode_slice"])
        expected = selection_keys[start:end]
        expected_set = set(expected)
        if not expected or len(expected) != len(expected_set):
            raise ValueError(f"Invalid selection slice in {source}: [{start}, {end})")

        rows = _read_jsonl(source / "episodes.jsonl")
        selected_rows = {}
        stale_rows = 0
        for row in rows:
            key = (str(row.get("scene", "")), str(row.get("episode_id", "")))
            if key not in expected_set:
                stale_rows += 1
                continue
            if key in selected_rows:
                raise ValueError(f"Duplicate collection row {key} in {source}")
            selected_rows[key] = row
        missing = expected_set - selected_rows.keys()
        if missing:
            raise ValueError(f"{source} is missing {len(missing)} selected episodes; "
                             f"first={sorted(missing)[:5]}")

        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("completed") != manifest.get("planned"):
            raise ValueError(f"Collection is incomplete: {source} "
                             f"({manifest.get('completed')}/{manifest.get('planned')})")
        if int(manifest.get("errors", 0)):
            raise ValueError(f"Collection has {manifest['errors']} errors: {source}")

        destination_root = source / "foundation_success_selected_episodes"
        destination_root.mkdir(parents=True, exist_ok=True)
        successes = 0
        success_transitions = 0
        for key in expected:
            row = selected_rows[key]
            if row.get("status") == "error":
                raise ValueError(f"Selected collection row failed: {source} {key}")
            if not (row.get("success") and row.get("saved")):
                continue
            if key in global_success_keys:
                raise ValueError(f"Duplicate successful selected key across collections: {key}")
            global_success_keys.add(key)
            source_file = source / row["file"]
            if not source_file.is_file():
                raise FileNotFoundError(source_file)
            destination = destination_root / key[0] / source_file.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_file, destination)
            successes += 1
            success_transitions += int(row.get("transitions", 0))

        audits.append(dict(collection=str(source), selection=str(selected_file),
                           episode_slice=[start, end], selected_episodes=len(expected),
                           selected_rows=len(selected_rows), success_episodes=successes,
                           success_transitions=success_transitions,
                           stale_rows_excluded=stale_rows))

    total = dict(selected_episodes=sum(row["selected_episodes"] for row in audits),
                 selected_success_episodes=sum(row["success_episodes"] for row in audits),
                 selected_success_transitions=sum(row["success_transitions"] for row in audits),
                 collections=audits)
    audit_path = Path(audit_path).expanduser().resolve()
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(total, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(total, indent=2), flush=True)
    return total


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", default="",
                        help="Fallback selection for collection runs without selection metadata")
    parser.add_argument("--collections", nargs="+", required=True)
    parser.add_argument("--audit", required=True)
    args = parser.parse_args(argv)
    prepare(args.collections, args.audit, args.selection or None)


if __name__ == "__main__":
    main()
