"""Print a bounded status summary, without streaming training/simulator logs."""
import argparse
import json
from pathlib import Path


def alive(pid):
    if not pid:
        return False
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def latest_json(path):
    if not path.exists():
        return None
    with path.open('rb') as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - 32768))
        lines = handle.read().decode(errors='replace').splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)
        except ValueError:
            continue


def status(root):
    output = {}
    state_file = root / 'pipeline_status.json'
    if state_file.exists():
        state = json.loads(state_file.read_text())
        output.update(stage=state.get('stage'), status=state['status'],
                      pipeline_alive=alive(state.get('pid')), child_alive=alive(state.get('child_pid')))
        if state.get('log'):
            row = latest_json(Path(state['log']))
            if row:
                output['latest'] = {key: row[key] for key in ('stage', 'split', 'epoch', 'episodes', 'total', 'completed', 'errors', 'status', 'SR', 'SPL') if key in row}
            if state['status'] == 'failed':
                output['failure_log'] = state['log']
    foundation = latest_json(root / 'logs/foundation.log')
    if foundation:
        output['foundation'] = {key: foundation[key] for key in ('stage', 'epoch', 'selection_score') if key in foundation}
    results = root / 'final_results.json'
    if results.exists():
        output['final_results'] = json.loads(results.read_text())['test']
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', default='result/2026-10-27-avnav-repaired')
    args = parser.parse_args()
    print(json.dumps(status(Path(args.run_dir)), ensure_ascii=False))
