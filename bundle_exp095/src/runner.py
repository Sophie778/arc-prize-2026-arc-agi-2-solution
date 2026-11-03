"""TP4 streaming measurement; reads no labels or NVARC predictions. MIT-0."""
import argparse
import asyncio
from collections.abc import Mapping
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import time
import traceback
from core import prompt, save

FORCE_PROMPT = "Stop analysing now. Apply the rule you were converging on and give the final test output grid(s)."


def prompt_tokens(tok, text, effort):
    tokens = tok.apply_chat_template([dict(role='user', content=text)], tokenize=True,
        return_dict=False, add_generation_prompt=True, enable_thinking=True, reasoning_effort=effort)
    if isinstance(tokens, Mapping): tokens = tokens['input_ids']
    tokens = list(tokens)
    if not tokens or not all(type(t) is int for t in tokens):
        raise ValueError('chat template must return nonempty list[int]')
    return tokens


def close_thought(thought, close, eos_ids, force_tokens=()):
    """Preserve stop evidence; remove terminal EOS before forced continuation."""
    generated = list(thought.token_ids)
    stop_reason = getattr(thought, 'stop_reason', None)
    natural = stop_reason == close[0] or bool(generated and generated[-1] == close[0])
    if natural:
        mode = 'natural'
    elif thought.finish_reason == 'length':
        mode = 'cap'
    elif thought.finish_reason == 'stop':
        mode = 'eos_before_delimiter'
    else:
        raise RuntimeError(f'unexpected thinking finish: {thought.finish_reason!r}')
    evidence = dict(natural_thinking_end=natural, thinking_finish_reason=thought.finish_reason,
        thinking_stop_reason=stop_reason, thinking_tail_token_ids=generated[-16:],
        thinking_capped=mode == 'cap', forced_close=not natural, thinking_close_reason=mode)
    continuation = generated[:]
    if not natural:
        while continuation and continuation[-1] in eos_ids:
            continuation.pop()
        continuation += list(force_tokens)
    evidence['force_prompt_inserted'] = not natural and bool(force_tokens)
    if not continuation or continuation[-1] != close[0]:
        continuation += close
    return continuation, evidence


def model_eos_ids(tok, model):
    config_path = Path(model)/'generation_config.json'
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    eos_ids = set()
    for value in [tok.eos_token_id, config.get('eos_token_id')]:
        eos_ids.update(value if isinstance(value, list) else [] if value is None else [value])
    return eos_ids


async def rolling(items, one, concurrency, deadline, event):
    iterator = iter(items); active = set(); exhausted = False
    try:
        while not exhausted or active:
            while len(active) < concurrency and not exhausted and time.monotonic() < deadline:
                item = next(iterator, None)
                if item is None: exhausted = True; break
                active.add(asyncio.create_task(one(item)))
                event('admit', item, len(active))
            if not active or time.monotonic() >= deadline: break
            done, active = await asyncio.wait(active, timeout=max(0, deadline-time.monotonic()),
                                               return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result(); event('release', None, len(active))
            if not done: break
    finally:
        for task in active: task.cancel()
        await asyncio.gather(*active, return_exceptions=True)


async def measure(a):
    from qwen_config import settings
    config = settings()
    print("Qwen settings: "+json.dumps(config), flush=True)
    versions={}
    for name in ['torch','transformers','vllm']:
        try:versions[name]=md.version(name)
        except md.PackageNotFoundError:versions[name]=None
    save(a.out/'engine_config.json',dict(versions=versions,cpu=a.cpu,model=a.model,
        tp=0 if a.cpu else 4,max_num_seqs=2 if a.cpu else 16,enforce_eager=False,
        configured_thinking_cap=a.thinking_cap,configured_answer_cap=a.answer_cap,
        mtp_requested=not a.disable_mtp, qwen=config))
    if not a.cpu and versions['vllm']!='0.19.1':
        print('WARNING: vLLM differs from reference 0.19.1; continuing',flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True, trust_remote_code=False)
    eos_ids = model_eos_ids(tok, a.model)
    tasks = json.loads((a.payload/'challenges.json').read_text())
    ids = json.loads((a.payload/('cpu_ids.json' if a.cpu else 'ids.json')).read_text())
    template = (a.payload/'system_prompt.txt').read_text()
    force_tokens = tok.encode('\n\n'+FORCE_PROMPT+'\n', add_special_tokens=False)
    suffix = tok.encode('\n\n', add_special_tokens=False)
    if a.cpu: tok.chat_template = (a.payload/'chat_template.jinja').read_text()
    close = tok.encode('</think>', add_special_tokens=False)
    if len(close) != 1: raise ValueError('expected one think-end token')
    prompt_max = max((len(prompt_tokens(tok, prompt(tasks[r['task']], r['idx'], template), config['effort'])) for r in ids), default=0)
    # Same 128K ceiling as xhigh65; oversized individual inputs retain the skip path.
    required = prompt_max + a.thinking_cap + a.answer_cap + len(force_tokens) + len(close) + len(suffix)
    context_limit = 32768 if a.cpu else min(131072, ((required+1023)//1024)*1024)
    save(a.out/'context_config.json', dict(prompt_max=prompt_max, required=required, max_model_len=context_limit, force_tokens=len(force_tokens)))
    if a.cpu:
        from tiny_engine import TinyEngine, Params
        engine = TinyEngine(a.model, tok)
        make_params = lambda **kw: Params(**kw)
    else:
        from vllm import SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM
        from vllm.sampling_params import RequestOutputKind
        engine = AsyncLLM.from_engine_args(AsyncEngineArgs(model=a.model, tensor_parallel_size=4,
            quantization='fp8', dtype='bfloat16', seed=0, max_model_len=context_limit,
            max_num_seqs=16, max_num_batched_tokens=8192, gpu_memory_utilization=.85,
            kv_cache_dtype=getattr(a, 'kv_dtype', None) or config['kv_cache_dtype'], enforce_eager=False, enable_prefix_caching=False,
            language_model_only=True, trust_remote_code=False,
            speculative_config=dict(method='mtp', num_speculative_tokens=3) if not a.disable_mtp else None,
            disable_log_stats=False))
        make_params = lambda **kw: SamplingParams(output_kind=RequestOutputKind.CUMULATIVE, **kw)
    ready = dict(monotonic=time.monotonic(), mtp_enabled=not a.cpu and not a.disable_mtp,
                 cpu=a.cpu, kv_cache_dtype=getattr(a, 'kv_dtype', None) or config['kv_cache_dtype'], concurrency=2 if a.cpu else config['concurrency'], max_model_len=context_limit, speculative_config=dict(method='mtp', num_speculative_tokens=3) if not a.disable_mtp else None)
    save(a.out/'engine_ready.json', ready)
    print('Qwen engine ready: '+json.dumps(ready), flush=True)
    if a.cpu:
        token_checks=[]
        for row in json.loads((a.payload/'ids.json').read_text()):
            text=prompt(tasks[row['task']], row['idx'], template)
            low=prompt_tokens(tok,text,config['effort'])
            token_checks.append(dict(key=row['key'],effort=config["effort"],prompt_tokens=len(low),context_fits=len(low)+49152+8192+len(force_tokens)+len(close)+len(suffix)<=131072))
        save(a.out/'tokenization.json',token_checks)
    events = []
    def event(kind, item, n):
        events.append(dict(kind=kind, job=item, active=n, monotonic=time.monotonic()))
        save(a.out/'scheduler.json', events)
    try:
        for cycle in range(3 if a.cpu else 1):
            wall = time.monotonic()
            save(a.out/f'generation_start_{cycle}.json',dict(cycle=cycle,monotonic=wall))
            stats_by = {arm: dict(arm=arm, cycle=cycle, generated_tokens=0, started=0, completed=0,
                all_four_gpus=not a.cpu) for arm in ['L']}
            async def one(job):
                row=job["row"]; arm=job["arm"]; effort=config["effort"]
                stats=stats_by[arm]
                start = time.monotonic(); request = f'{cycle}_{arm}_{row["key"]}'
                path = a.out/'records'/f'{request}.json'
                r = dict(key=row['key'], task=row['task'], idx=row['idx'], arm=arm, cycle=cycle,
                    effort=effort, thinking_cap=a.thinking_cap, thinking_tokens=0, answer_tokens=0,
                    thinking_text='', answer_text='', natural_thinking_end=None,
                    natural_end=None, complete=False, seconds=0., stage='tokenizing')
                stats['started'] += 1
                async def stage(inp, cap, thinking):
                    last = None; previous = 0; saved = 0
                    name = 'thinking' if thinking else 'answer'
                    params = make_params(temperature=1., top_p=.95, top_k=20, seed=0,
                        max_tokens=cap, stop_token_ids=close if thinking else None,
                        skip_special_tokens=False)
                    try:
                        async for output in engine.generate({'prompt_token_ids':inp}, params, request+'_'+name):
                            last = output.outputs[0]; count = len(last.token_ids)
                            stats['generated_tokens'] += count-previous; previous = count
                            r[name+'_tokens'] = count
                            # JSON escaping uses <=6 bytes/character, hence <=1.5MiB per field.
                            r[name+'_text'] = last.text[:256*1024]
                            r[name+'_text_truncated'] = len(last.text)>256*1024
                            if count-saved >= 2048 or output.finished:
                                r['seconds'] = time.monotonic()-start; save(path, r); saved = count
                                save(a.out/f'meter_{cycle}_{arm}.json',stats)
                        return last
                    finally:
                        await engine.abort(request+'_'+name)
                try:
                    text = prompt(tasks[row['task']], row['idx'], template)
                    tokens = prompt_tokens(tok, text, effort)
                    r.update(prompt_tokens=len(tokens), prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                             tokenized_prompt_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest())
                    if len(tokens)+a.thinking_cap+a.answer_cap+len(force_tokens)+len(close)+len(suffix) > context_limit:
                        r.update(answer_skipped=True, skip_reason='input_plus_generation_exceeds_context',
                                 complete=True, stage='skipped_context')
                        return
                    r['stage'] = 'thinking'; save(path, r)
                    thought = await stage(tokens, a.thinking_cap, True)
                    if thought is None: raise RuntimeError('no thinking output')
                    generated, evidence = close_thought(thought, close, eos_ids, force_tokens)
                    r.update(evidence)
                    print('Qwen thinking closed: '+json.dumps(dict(key=row['key'], thinking_tokens=r['thinking_tokens'], **evidence)), flush=True)
                    natural = evidence['natural_thinking_end']
                    continuation = tokens + generated
                    continuation += suffix
                    r['stage'] = 'answer'; save(path, r)
                    answer = await stage(continuation, a.answer_cap, False)
                    if answer is None: raise RuntimeError('no answer output')
                    r.update(answer_finish_reason=answer.finish_reason,
                        natural_end=natural and answer.finish_reason == 'stop', complete=True, stage='finished')
                    stats['completed'] += 1
                except asyncio.CancelledError:
                    r.update(stage='deadline', deadline_stopped=True); raise
                except Exception as exc:
                    r.update(error=repr(exc), traceback=traceback.format_exc(), stage='error')
                finally:
                    r['seconds'] = time.monotonic()-start; save(path, r)
            jobs=[dict(row=row,arm=arm) for row in ids for arm in ['L']]
            try:
                await rolling(jobs, one, 2 if a.cpu else config["concurrency"], a.deadline, event)
            finally:
                elapsed=time.monotonic()-wall
                for arm,stats in stats_by.items():
                    stats['generation_wall_seconds']=elapsed
                    stats['aggregate_tok_s']=stats['generated_tokens']/max(elapsed,1e-9)
                    stats['throughput_definition']='arm tokens / shared L generation wall seconds; total GPU tok/s'
                    save(a.out/f'throughput_{cycle}_{arm}.json',stats)
            if time.monotonic() >= a.deadline: break
    finally:
        import core
        save(a.out/'runner_writes.json', dict(cumulative_json_written_bytes=core.WRITTEN_BYTES))
        engine.shutdown()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--payload', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model', required=True); p.add_argument('--deadline', type=float, required=True)
    p.add_argument('--cpu', action='store_true'); p.add_argument('--thinking-cap', type=int, default=49152)
    p.add_argument('--disable-mtp', action='store_true')
    p.add_argument('--answer-cap', type=int, default=8192)
    p.add_argument('--kv-dtype', choices=['fp8', 'auto'])
    asyncio.run(measure(p.parse_args()))
