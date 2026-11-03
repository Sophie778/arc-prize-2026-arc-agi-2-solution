"""Per-task durable starts/ends, without changing scheduling. MIT-0."""
import json
import os
from pathlib import Path
import time
from core import save

def mark(key, phase, rank):
    root = Path(os.environ['COMBO_TASK_TRACE'])
    save(root/f'{key}.{phase}.json', dict(task=key, phase=phase, rank=rank,
         wall=time.time(), monotonic=time.monotonic()))
    print(f'[task] {key} {phase} wall={time.time()} monotonic={time.monotonic()}', flush=True)

def summary(tasks, root, start):
    records = []
    for path in sorted(root.glob('*.json')):
        try: records.append(json.loads(path.read_text()))
        except (OSError, ValueError): pass
    started = {r['task'] for r in records if r['phase'] == 'start'} & set(tasks)
    finished = {r['task'] for r in records if r['phase'] == 'end'} & set(tasks)
    return dict(unstarted_tasks=len(set(tasks)-started), unfinished_tasks=len(set(tasks)-finished),
                in_progress_tasks=len(started-finished), records=[dict(r, elapsed=r['monotonic']-start) for r in records])
