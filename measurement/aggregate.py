"""Whitelist-only export of historical S3 metrics; standard library, MIT-0.

Reads raw private results but never exports identifiers, hashes of prompts,
text, token IDs, grids, filenames, paths, tracebacks or correctness labels.
Rows are sorted by retained metrics, with no cross-condition request linkage.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

REQUEST_FIELDS = ('prompt_tokens', 'thinking_tokens', 'answer_tokens', 'complete',
    'natural_thinking_end', 'natural_end', 'thinking_capped', 'forced_close',
    'thinking_close_reason', 'thinking_finish_reason', 'answer_finish_reason',
    'deadline_stopped', 'seconds', 'stage')
STAGES = {'tokenizing', 'thinking', 'answer', 'finished', 'deadline', 'error'}
REASONS = {'natural', 'cap', 'eos_before_delimiter'}
FINISH = {'stop', 'length', 'abort', 'error'}


def aggregate(root):
    root = Path(root)
    original = json.loads((root / 'result.json').read_text())
    result = []
    for condition in original['conditions']:
        arm = condition['condition']
        if arm not in ('B', 'E', 'F'):
            raise ValueError('unknown condition')
        raw = [json.loads(p.read_text()) for p in (root / arm / 'records').glob('*.json')]
        rows = [{k: r.get(k) for k in REQUEST_FIELDS} for r in raw]
        for row in rows:
            for k, v in row.items():
                allowed = STAGES if k == 'stage' else REASONS if k == 'thinking_close_reason' else FINISH if k.endswith('finish_reason') else None
                if allowed is not None:
                    if v is not None and v not in allowed:
                        raise ValueError('unexpected categorical metric')
                elif v is not None and type(v) not in (int, float, bool):
                    raise ValueError('non-numeric metric')
        rows.sort(key=lambda r: json.dumps(r, sort_keys=True))
        complete = sum(bool(r['complete']) for r in rows)
        if complete != condition['completed']:
            raise ValueError('completion mismatch')
        closed = Counter(r['thinking_close_reason'] for r in rows if r['complete'])
        progress = condition['progress']
        windows = [{k: w[k] for k in ('start_seconds', 'end_seconds', 'generated_tokens', 'tok_s', 'complete_window')} for w in progress['windows']]
        result.append(dict(condition=arm, expected_requests=32, observed_requests=len(rows),
            completed=complete, unfinished=32-complete, not_started=32-len(rows),
            complete_by_thinking_close_reason=dict(sorted(closed.items())),
            startup_wall_seconds=condition['startup_wall_seconds'],
            condition_wall_seconds=condition['seconds'],
            generated_tokens=progress['generated_tokens'],
            generation_wall_seconds=progress['generation_wall_seconds'],
            aggregate_tok_s=progress['aggregate_tok_s'], windows=windows,
            request_metrics=rows))
    return dict(conditions=result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    data = json.dumps(aggregate(a.input), indent=2, allow_nan=False) + '\n'
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(data)


if __name__ == '__main__':
    main()
