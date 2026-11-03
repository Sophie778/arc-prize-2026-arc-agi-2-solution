"""exp094 pure scheduling and answer policy. MIT-0."""
from copy import deepcopy
from dataclasses import dataclass, field
from experiment_config import nvarc_settings
import hashlib
from core import extract, valid_grid

# Chosen once, without labels or observed successes. Never tune for evaluation IDs.
SALT = 'arc2-exp091-qwen-combo-20261024-v1'


@dataclass(frozen=True)
class Budget:
    nvarc_stop: float = field(default_factory=lambda: nvarc_settings()["stop"])
    nvarc_hard: float = None

    def __post_init__(self):
        if self.nvarc_hard is None:
            object.__setattr__(self, "nvarc_hard", self.nvarc_stop + 300)
    qwen_stop: float = 41700        # 11h35
    final_save: float = 42000       # 11h40
    minimum_qwen: float = 3900      # 65m of the 12h total remains
    total: float = 43200

    @classmethod
    def commit(cls):
        return cls(2400, 2520, 3420, 3510, 600, 3600)

    def can_start_qwen(self, elapsed):
        return elapsed < self.qwen_stop and self.total - elapsed >= self.minimum_qwen

    def phase(self, elapsed):
        if elapsed < self.nvarc_stop: return 'nvarc'
        if elapsed < self.nvarc_hard: return 'nvarc_cleanup'
        if elapsed < self.qwen_stop: return 'qwen'
        if elapsed < self.final_save: return 'final_save'
        return 'done'


def select(tasks, limit=None, candidate_counts=None):
    rows = [dict(key=f'{task}_{idx}', task=task, idx=idx)
            for task in tasks for idx in range(len(tasks[task]['test']))]
    # Missing/zero candidates share the first tier; the original salt orders each tier.
    counts = candidate_counts
    return sorted(rows, key=lambda r: (
        0 if counts is None or counts.get(r['key'], 0) == 0 else 1, hashlib.sha256(
        (SALT + ':' + r['key']).encode()).hexdigest(), r['key']))[:limit]


def initial(tasks):
    return {key: [dict(attempt_1=[[0]], attempt_2=[[0]]) for _ in task['test']]
            for key, task in tasks.items()}


def validate(submission, tasks):
    if set(submission) != set(tasks): raise ValueError('submission IDs differ')
    for task, data in tasks.items():
        rows = submission[task]
        if not isinstance(rows, list) or len(rows) != len(data['test']):
            raise ValueError('test count differs: ' + task)
        for row in rows:
            if set(row) != {'attempt_1', 'attempt_2'} or not all(valid_grid(g) for g in row.values()):
                raise ValueError('invalid attempts: ' + task)
    return submission


def merge_record(submission, baseline, record, allowed, candidate_counts=None):
    """Return a new submission, or unchanged copy. Never trust record task/idx."""
    result = deepcopy(submission)
    row = allowed.get(record.get('key'))
    if row is None: return result, False
    if not (record.get('complete') is True
            and record.get('answer_finish_reason') == 'stop'
            and not record.get('answer_text_truncated') and not record.get('error')
            and not record.get('deadline_stopped')):
        return result, False
    grid = extract(record.get('answer_text', ''))
    task, idx = row['task'], row['idx']
    if grid is None: return result, False
    first = baseline[task][idx]['attempt_1']
    present = (candidate_counts.get(record['key'], 0) > 0 if candidate_counts is not None
               else first != [[0]])
    if not present:
        result[task][idx]['attempt_1'] = grid
    elif grid != first:
        result[task][idx]['attempt_2'] = grid
    else:
        return result, False
    return result, True
