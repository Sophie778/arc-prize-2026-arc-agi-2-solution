"""Measured FP8 KV and concurrency 16 defaults; optional BF16/auto fallback. MIT-0."""
import os

def settings(env=None):
    env = os.environ if env is None else env
    kv = env.get('QWEN_KV', 'fp8')
    if kv not in ('bf16', 'fp8'): raise ValueError('QWEN_KV must be bf16 or fp8')
    conc = int(env.get('QWEN_CONC', '16'))
    if not 1 <= conc <= 16: raise ValueError('unsafe QWEN_CONC')
    return dict(kv=kv, kv_cache_dtype='auto' if kv == 'bf16' else 'fp8',
                concurrency=conc, max_num_seqs=16, max_model_len=131072,
                effort='xhigh', thinking_cap=49152, answer_cap=8192)
