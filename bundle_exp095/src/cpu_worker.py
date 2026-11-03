"""Model-free smoke worker using real task inputs; only submission plumbing. MIT-0."""
import argparse
import json
from pathlib import Path
from core import save
from policy import select

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path); p.add_argument('--tasks', type=Path)
    a = p.parse_args(); tasks = json.loads(a.tasks.read_text())
    for i, row in enumerate(select(tasks, 2)):
        grid = tasks[row['task']]['test'][row['idx']]['input']
        # Make duplicate detection deterministic even if the input is the 1x1 fallback.
        if grid == [[0]]: grid = [[1]]
        save(a.out/'qwen/records'/f"0_L_{row['key']}.json", dict(row, complete=True,
             natural_thinking_end=i==0, forced_close=i!=0, force_prompt_inserted=i!=0,
             thinking_close_reason='natural' if i==0 else 'cap', thinking_tokens=2048 if i else 10, answer_tokens=10,
             answer_finish_reason='stop', answer_text=json.dumps(grid)))
