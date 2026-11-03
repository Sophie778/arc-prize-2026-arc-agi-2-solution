"""Measured FP8 KV and concurrency 16 defaults; optional BF16/auto fallback. MIT-0."""
import os
from experiment_config import QWEN_CONC

def settings(env=None):
    env = os.environ if env is None else env
    kv = env.get('QWEN_KV', 'fp8')
    if kv != 'fp8': raise ValueError('exp096 requires FP8 KV')
    conc = int(env.get('QWEN_CONC', QWEN_CONC))
    if conc not in (16, 32): raise ValueError('unsafe QWEN_CONC')
    return dict(kv=kv, kv_cache_dtype='auto' if kv == 'bf16' else 'fp8',
                concurrency=conc, max_num_seqs=conc, max_model_len=131072,
                effort='xhigh', thinking_cap=49152, answer_cap=8192)
