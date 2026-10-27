"""Retry only train collision tasks that exhausted the first collection budget."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiment.avnav.collect_expert_diverse_collision_data import (
    MAX_COLLISION_ATTEMPTS,
    _collect_one,
    _load_plan,
    _merge_manifests,
    _setup_env,
    _write_atomic,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def repair(output_root, plan_path, workers=4, attempts_per_task=30):
    output_root = Path(output_root).expanduser().resolve()
    plan_path = Path(plan_path).expanduser().resolve()
    category_root = output_root / "collision"
    all_tasks = [json.loads(line) for line in plan_path.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    all_tasks = [row for row in all_tasks if row["category"] == "collision"]
    shortfall_rows = [row for row in _jsonl(category_root / "manifest.jsonl")
                      if row.get("status") == "shortfall"]
    shortfall_ids = {row["task_id"] for row in shortfall_rows}
    if not shortfall_ids:
        return {"status": "no_shortfalls", "repaired": 0}
    task_by_id = {row["task_id"]: row for row in all_tasks}
    if not shortfall_ids <= task_by_id.keys():
        raise ValueError("Shortfall manifest contains tasks absent from the frozen plan")

    tasks_by_worker = {worker: [] for worker in range(workers)}
    worker_for_task = {}
    for index, task in enumerate(all_tasks):
        worker = index % workers
        worker_for_task[task["task_id"]] = worker
        if task["task_id"] in shortfall_ids:
            tasks_by_worker[worker].append(task)

    starting_attempts = {}
    for worker, tasks in tasks_by_worker.items():
        attempts_path = category_root / f"attempts_worker_{worker:02d}.jsonl"
        rows = _jsonl(attempts_path)
        last_by_task = {}
        for row in rows:
            task_id = row.get("task_id")
            if task_id in shortfall_ids and isinstance(row.get("attempt"), int):
                last_by_task[task_id] = max(last_by_task.get(task_id, -1), row["attempt"])
        for task in tasks:
            task_id = task["task_id"]
            starting_attempts[task_id] = last_by_task.get(task_id, -1) + 1

    repair_protocol_path = output_root / "collision_repair_protocol.json"
    frozen = {
        "status": "frozen_before_repair",
        "reason": "six train-only collision tasks had no collision after the original 30 attempts",
        "output_root": str(output_root),
        "selection_manifest": str(plan_path),
        "selection_manifest_sha256": _sha256(plan_path),
        "initial_collision_manifest_sha256": _sha256(category_root / "manifest.jsonl"),
        "attempts_per_shortfall_task": int(attempts_per_task),
        "shortfall_tasks": [
            {"task_id": task_id, "scene": task_by_id[task_id]["scene"],
             "episode_id": task_by_id[task_id]["episode_id"],
             "parent_replicate": int(task_by_id[task_id]["parent_replicate"]),
             "worker": worker_for_task[task_id],
             "first_new_attempt": starting_attempts[task_id]}
            for task_id in sorted(shortfall_ids)
        ],
        "action_handling": (
            "same frozen collision collection policy: fixed first-eight-step exploration "
            "schedule independent of collisions, then native raw oracle actions; no "
            "collision-conditioned action override or forced recovery"
        ),
        "reward_shaping": "unchanged from collection_protocol.json",
        "data_split": "train_multiple only; no validation or test data used",
    }
    if repair_protocol_path.exists():
        if json.loads(repair_protocol_path.read_text(encoding="utf-8")) != frozen:
            raise FileExistsError("Existing collision repair protocol does not match")
    else:
        repair_protocol_path.write_text(
            json.dumps(frozen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    env, episodes = _setup_env(list(task_by_id[task_id] for task_id in sorted(shortfall_ids)))
    repaired_rows = []
    try:
        for worker in sorted(tasks_by_worker):
            for task in tasks_by_worker[worker]:
                task_id = task["task_id"]
                episode = episodes[(str(task["scene"]), str(task["episode_id"]))]
                first = starting_attempts[task_id]
                final_row = None
                attempts_path = category_root / f"attempts_worker_{worker:02d}.jsonl"
                for attempt in range(first, first + attempts_per_task):
                    try:
                        trajectory = _collect_one(env, episode, task, attempt)
                        row = {
                            "task_id": task_id, "category": "collision",
                            "scene": task["scene"], "episode_id": task["episode_id"],
                            "attempt": attempt, "steps": len(trajectory["action_id"]),
                            "collisions": trajectory["collision_count"],
                            "recoveries": trajectory["recovery_event_count"],
                            "unresolved": trajectory["unresolved_collision_count"],
                            "success": trajectory["episode_success"],
                            "spl": trajectory["spl"], "error": None,
                        }
                        if trajectory["collision_count"]:
                            out_path = (category_root / "episodes" / str(task["scene"])
                                        / f"{task_id}.pkl")
                            _write_atomic(out_path, trajectory)
                            row.update(status="saved", file=str(out_path))
                            final_row = row
                            with attempts_path.open("a", encoding="utf-8") as handle:
                                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                                handle.flush()
                            break
                        with attempts_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                            handle.flush()
                        final_row = row
                    except Exception as error:  # retain per-attempt diagnostics
                        final_row = {
                            "task_id": task_id, "category": "collision",
                            "scene": task["scene"], "episode_id": task["episode_id"],
                            "attempt": attempt, "status": "error",
                            "error": f"{type(error).__name__}: {error}",
                        }
                        with attempts_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(final_row, ensure_ascii=False) + "\n")
                            handle.flush()
                if final_row is None or final_row.get("status") != "saved":
                    final_row = {
                        "task_id": task_id, "category": "collision",
                        "scene": task["scene"], "episode_id": task["episode_id"],
                        "status": "shortfall", "attempt": first + attempts_per_task - 1,
                        "reason": "no collision in additional repair budget",
                    }
                repaired_rows.append(final_row)

                worker_manifest = category_root / f"manifest_worker_{worker:02d}.jsonl"
                prior = [row for row in _jsonl(worker_manifest)
                         if row.get("task_id") != task_id]
                prior.append(final_row)
                temporary = worker_manifest.with_suffix(".jsonl.tmp")
                temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                                      for row in prior), encoding="utf-8")
                temporary.replace(worker_manifest)
    finally:
        env.close()

    summary = _merge_manifests(output_root, "collision", workers)
    report_path = output_root / "collection_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["category_summaries"] = [
        summary if row["category"] == "collision" else row
        for row in report["category_summaries"]
    ]
    report["status"] = ("completed" if all(
        row["errors"] == 0 and row["shortfalls"] == 0
        for row in report["category_summaries"]
    ) else "completed_with_shortfalls_or_errors")
    report["total_target_episodes"] = sum(
        row["target_tasks"] for row in report["category_summaries"])
    report["total_saved_episodes"] = sum(
        row["saved_episodes"] for row in report["category_summaries"])
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    temporary.replace(report_path)

    frozen["status"] = "repair_completed"
    frozen["repair_attempts"] = repaired_rows
    frozen["result_manifest"] = str((category_root / "manifest.jsonl").resolve())
    frozen["result_manifest_sha256"] = _sha256(category_root / "manifest.jsonl")
    repair_protocol_path.write_text(
        json.dumps(frozen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"status": report["status"], "repaired_tasks": repaired_rows,
            "collision_summary": summary, "collection_report": str(report_path)}


if __name__ == "__main__":
    result = repair("/data2/getuanhui/avnav-codex",
                    "/data2/getuanhui/avnav-codex/selection_manifest.jsonl",
                    workers=4, attempts_per_task=30)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
