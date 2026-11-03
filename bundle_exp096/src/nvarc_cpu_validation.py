"""Exercise production view batching and exact DFS functions without Unsloth. MIT-0."""
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

from arc_loader import ArcDataset, QwenFormatter
from generation_views import batches_for_outputs

H = Path(__file__).resolve().parent


def tokenizer():
    from transformers import AutoTokenizer
    root = H/'nvarc_tokenizer'
    provenance = json.loads((root/'provenance.json').read_text())
    for name, sha in provenance['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == sha
    return AutoTokenizer.from_pretrained(root, local_files_only=True)


def dfs_entry():
    # Execute the unchanged production DFS definitions, bypassing training-only imports.
    import torch
    tree = ast.parse((H/'arc_solver.py').read_text())
    names = {'ARC_VOCAB', 'ARC_TOKENS', 'PAD_ID', 'EOS_ID', 'turbo_dfs', 'inference_turbo_dfs'}
    nodes = [n for n in tree.body if
             (isinstance(n, ast.FunctionDef) and n.name in names) or
             (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets))]
    scope = dict(torch=torch, time=time, defaultdict=defaultdict)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(H/'arc_solver.py'), 'exec'), scope)
    return scope['inference_turbo_dfs']


def tiny_model():
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.set_num_threads(2)
    torch.manual_seed(0)
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=16, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=8192, eos_token_id=15, pad_token_id=13)).eval()


def prepare(task_id, task, tok, max_len=None):
    fmt = QwenFormatter(tok)
    ds = ArcDataset(queries={task_id: task}).split_multi_replies().augment(n=1, seed=2)
    ds = ds.cut_to_len(formatter=fmt, name='input', max_len=(8192-fmt.max_new_tokens() if max_len is None else max_len))
    groups = {}
    for key in sorted(ds.keys):
        groups.setdefault(key.split('.')[0], []).append(key)
    tokens = {k: tok.encode(ds.get(k, fmt)['input']) for k in ds.keys}
    return ds, groups, tokens, batches_for_outputs(groups, tokens)


def run(tasks, cycles=3):
    tok = tokenizer()
    model, inference = tiny_model(), dfs_entry()
    checks = []
    for cycle in range(cycles):
        for key, task in tasks.items():
            ds, groups, tokens, batches = prepare(key, task, tok)
            flat = sum(batches, [])
            assert len(flat) == len(set(flat)) == 8*len(task['test'])
            assert set(flat) == set(ds.keys)
            lengths = []
            for batch in batches:
                sizes = {len(tokens[k]) for k in batch}
                assert len(sizes) == 1 and 1 <= len(batch) <= 4
                # A real tiny Qwen forward through the production torch.tensor and DFS.
                result = inference(model, [tokens[k] for k in batch], max_new_tokens=1,
                                   max_score=100, end_time=time.time()+60)
                assert {i for i, _ in result} == set(range(len(batch)))
                assert all(beams and beams[0][1] == [15] for _, beams in result)
                lengths.append(dict(views=len(batch), tokens=next(iter(sizes))))
            checks.append(dict(cycle=cycle, task=key, outputs=len(groups), views=len(flat),
                               rectangular=any(len(t['input']) != len(t['input'][0]) for t in task['test']),
                               batches=lengths, dfs_batches=len(batches)))
    return dict(tokenizer=json.loads((H/'nvarc_tokenizer/provenance.json').read_text()),
                cycles=cycles, checks=checks, dfs_batches=sum(c['dfs_batches'] for c in checks))
