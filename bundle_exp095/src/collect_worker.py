"""Bounded CPU candidate recovery child. No CUDA initialization. MIT-0."""
import json
from pathlib import Path
import sys
from core import save
from supervisor import collect

if __name__ == '__main__':
    tasks, store, target = map(Path, sys.argv[1:])
    counts = {}
    submission = collect(json.loads(tasks.read_text()), store, counts)
    save(target.with_name('recovered_counts.json'), counts)
    save(target, submission)
