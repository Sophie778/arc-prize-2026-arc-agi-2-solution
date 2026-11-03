"""CUDA-free parent: recover NVARC, release GPUs, then run isolated TP4. MIT-0."""
import bz2
from copy import deepcopy
import ctypes
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import threading
import time

from core import save, valid_grid
from policy import Budget, initial, merge_record, select, validate


def stop(p):
    """Kill the whole session, including workers after the session leader exits."""
    if p is None: return
    import psutil
    descendants = psutil.Process().children(recursive=True)
    try: os.killpg(p.pid, signal.SIGTERM)
    except ProcessLookupError: pass
    try: p.wait(timeout=3)
    except subprocess.TimeoutExpired: pass
    for child in descendants:
        try: child.kill()
        except psutil.NoSuchProcess: pass
    try: os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError: pass
    p.wait(timeout=3)
    # Linux subreaper adopts grandchildren even when a worker exits early.
    children = psutil.Process().children(recursive=True)
    for child in children:
        try: child.kill()
        except psutil.NoSuchProcess: pass
    psutil.wait_procs(children, timeout=3)
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
            if pid == 0: break
        except ChildProcessError: break


def gpus_free():
    """Require four idle devices and no compute processes; uncertainty skips Qwen."""
    try:
        mem = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3, check=True)
        rows = [line.split(',') for line in mem.stdout.strip().splitlines()]
        apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3, check=True)
        return (len(rows) == 4 and {int(r[0]) for r in rows} == set(range(4))
                and all(int(r[1]) <= 256 for r in rows) and not apps.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError): return False


def model_path(root=Path('/kaggle/input'), explicit=None):
    """Only the measured official FP8 mount/version is accepted, old or new."""
    candidates = [Path(explicit)] if explicit else sorted({p.parent for p in root.rglob('config.json')
        if 'qwen3-8-27b-fp8-official' in p.parts and p.parent.parts[-3:] == ('pytorch', 'hf-fp8', '1')})
    valid = []
    for path in candidates:
        try:
            config = json.loads((path/'config.json').read_text())
            text = config.get('text_config', config)
            quant = config.get('quantization_config', {})
            if (config.get('model_type') == 'qwen3_5' and text.get('hidden_size') == 5120
                    and text.get('num_hidden_layers') == 64 and quant.get('quant_method') == 'fp8'
                    and (path/'tokenizer_config.json').is_file()
                    and ((path/'tokenizer.json').is_file())
                    and (path/'model.safetensors.index.json').is_file()):
                index = json.loads((path/'model.safetensors.index.json').read_text())
                shards = set(index.get('weight_map', {}).values())
                if shards and all((path/s).is_file() for s in shards): valid.append(path)
        except (OSError, ValueError): pass
    if len(valid) != 1: raise RuntimeError(f'expected one verified official FP8 model, got {valid}')
    return str(valid[0])


def collect(tasks, store, metadata=None):
    # ArcDecoder uses kgmon and insertion-order tie breaking from exp085.
    from arc_decoder import ArcDecoder
    decoder = ArcDecoder(None, 2)
    store.mkdir(parents=True, exist_ok=True)
    decoder.load_decoded_results(str(store))
    if metadata is not None:
        metadata.update({key: sum(valid_grid(sample['solution'].tolist()) for sample in samples.values())
                        for key, samples in decoder.decoded_results.items()})
    submission = initial(tasks)
    for key, guesses in decoder.run_selection_algo().items():
        task, idx = key.rsplit('_', 1)
        if task not in tasks or not idx.isdigit() or int(idx) >= len(tasks[task]['test']): continue
        for i, grid in enumerate(guesses[:2]):
            grid = grid.tolist()
            if valid_grid(grid): submission[task][int(idx)][f'attempt_{i+1}'] = grid
    return validate(submission, tasks)


def collect_bounded(tasks, store, payload, tmp, deadline):
    """A corrupt/in-progress candidate cannot hold up the parent or GPU cleanup."""
    remaining = deadline-time.monotonic()
    if remaining <= 0: raise TimeoutError('candidate recovery deadline')
    target = tmp/'recovered.json'
    subprocess.run([sys.executable, str(payload/'collect_worker.py'),
        str(tmp/'challenges.json'), str(store), str(target)],
        cwd=payload, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        timeout=min(20, remaining), check=True, start_new_session=True)
    return validate(json.loads(target.read_text()), tasks)


class Jobs:
    def __init__(self, env, tmp, out, start, clock=time.monotonic, sleep=time.sleep):
        self.env, self.tmp, self.out, self.start = env, tmp, out, start
        self.clock, self.sleep = clock, sleep
        self.p = None
        self.thread = None

    def launch(self, cmd, name, cwd=None):
        self.p = subprocess.Popen(cmd, env=self.env, cwd=cwd or self.tmp,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        def drain():
            written = 0
            with (self.out/name).open('wb') as stream:
                while True:
                    block = self.p.stdout.read(8192)
                    if not block: break
                    kept = block[:max(0, 128*1024**2-written)]
                    stream.write(kept); written += len(kept)
        self.thread = threading.Thread(target=drain, daemon=True)
        self.thread.start()

    def cleanup(self):
        stop(self.p)
        if self.thread: self.thread.join(timeout=5)
        self.p = self.thread = None

    def run(self, cmd, name, deadline):
        self.launch(cmd, name)
        try: return self.p.wait(timeout=max(.01, self.start+deadline-self.clock()))
        except subprocess.TimeoutExpired: return None
        finally: self.cleanup()


def run_qwen(jobs, py, payload, out, model, start, budget, rerun,
             allowed, merge_available, event, clock=time.monotonic, sleep=time.sleep):
    """Retry startup once in a fresh process, after killing/reaping TP workers."""
    ready_path = out/'qwen/engine_ready.json'
    from qwen_config import settings
    config = settings(jobs.env if hasattr(jobs, 'env') else None)
    kv, conc = config['kv_cache_dtype'], config['concurrency']
    # Startup failure at 32 gets exactly one fresh-process step down to 16.
    plans = ([(kv, False, 32), (kv, False, 16)] if conc == 32 else
             [(kv, False, 16), (kv, True, 16)])
    for attempt, (effective_kv, disable_mtp, effective_conc) in enumerate(plans):
        ready_path.unlink(missing_ok=True)
        cmd = [py, '-u', str(payload/'runner.py'), '--payload', str(payload),
               '--out', str(out/'qwen'), '--model', model,
               '--deadline', str(start+budget.qwen_stop-10),
               '--thinking-cap', '49152' if rerun else '2048', '--answer-cap', '8192', '--kv-dtype', effective_kv,
               '--concurrency', str(effective_conc)]
        if disable_mtp: cmd.append('--disable-mtp')
        jobs.launch(cmd, f'qwen_conc{effective_conc}.log' if conc == 32 else 'qwen_auto.log' if effective_kv != kv else 'qwen_no_mtp.log' if disable_mtp else 'qwen.log')
        event('qwen_started', selected=list(allowed), mtp_requested=not disable_mtp, kv_cache_dtype=effective_kv, concurrency=effective_conc, attempt=attempt+1)
        while jobs.p.poll() is None and clock()-start < budget.qwen_stop:
            merge_available(); sleep(1)
        rc = jobs.p.poll()
        jobs.cleanup()
        merge_available()
        ready = json.loads(ready_path.read_text()) if ready_path.exists() else None
        event('qwen_finished', returncode=rc, failure=rc != 0, attempt=attempt+1,
              engine_ready=ready, mtp_started=bool(ready and ready.get('mtp_enabled')))
        if attempt == len(plans)-1 or rc in (0, None) or ready is not None: break
        if clock()-start >= budget.qwen_stop-10 or not gpus_free():
            event('qwen_retry_skipped', reason='deadline or GPU release unverified'); break
        event('qwen_retry_concurrency16' if conc == 32 else 'qwen_retry_without_mtp' if attempt == 0 else 'qwen_retry_auto_kv',
              reason='startup failed; retry in fresh process', concurrency=plans[attempt+1][2])


def orchestrate(tasks, payload, out, tmp, start, cpu=False, jobs=None, clock=time.monotonic, sleep=time.sleep):
    rerun = bool(os.environ.get('KAGGLE_IS_COMPETITION_RERUN'))
    budget = Budget() if rerun else Budget.commit()
    submission = initial(tasks)
    validate(submission, tasks); save(out/'submission.json', submission)
    baseline = deepcopy(submission)
    status = dict(cpu=cpu, rerun=rerun, budget=budget.__dict__, events=[])
    def event(name, **kw):
        status['events'].append(dict(name=name, elapsed=clock()-start, **kw))
        save(out/'combo_status.json', status)
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        HF_HUB_DISABLE_TELEMETRY='1', TMPDIR=str(tmp), PIP_NO_INDEX='1', PIP_NO_CACHE_DIR='1',
        PYTHONDONTWRITEBYTECODE='1', XDG_CACHE_HOME=str(tmp/'cache'), HF_HOME=str(tmp/'hf'),
        TORCHINDUCTOR_CACHE_DIR=str(tmp/'inductor'), TRITON_CACHE_DIR=str(tmp/'triton'),
        VLLM_CACHE_ROOT=str(tmp/'vllm'), CUDA_CACHE_PATH=str(tmp/'cuda'),
        TOKENIZERS_PARALLELISM='false', VLLM_USE_DEEP_GEMM='0', VLLM_NO_USAGE_STATS='1',
        VLLM_WORKER_MULTIPROC_METHOD='spawn', UNSLOTH_DISABLE_STATISTICS='1', COMBO_TASK_TRACE=str(tmp/'task_trace'),
        TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas', OMP_NUM_THREADS='12')
    jobs = jobs or Jobs(env, tmp, out, start, clock, sleep)
    store = Path('/kaggle/inference_outputs')
    allowed = {r['key']: r for r in select(tasks, None if rerun else 2)}
    processed = set()
    candidate_counts = None
    def merge_available():
        nonlocal submission
        for path in sorted((out/'qwen/records').glob('*.json')):
            if path.name in processed: continue
            try: record = json.loads(path.read_text())
            except (OSError, ValueError): continue
            if not record.get('complete'): continue
            updated, accepted = merge_record(submission, baseline, record, allowed, candidate_counts)
            validate(updated, tasks)
            if accepted:
                save(out/'submission.json', updated)
                submission = updated
                event('qwen_accepted', key=record['key'], forced_close=record.get('forced_close'),
                      force_prompt_inserted=record.get('force_prompt_inserted'),
                      thinking_close_reason=record.get('thinking_close_reason'),
                      thinking_tokens=record.get('thinking_tokens'), answer_tokens=record.get('answer_tokens'))
            processed.add(path.name)
    try:
        event('initial_saved', outputs=sum(len(t['test']) for t in tasks.values()))
        if cpu:
            # Exercise the identical subprocess/record/atomic-save path, with no model imports.
            jobs.launch([sys.executable, str(payload/'cpu_worker.py'), '--out', str(out),
                         '--tasks', str(tmp/'challenges.json')], 'cpu.log')
            while jobs.p.poll() is None: sleep(.05)
            if jobs.p.returncode: raise RuntimeError('CPU worker failed')
            jobs.cleanup()
            save(out/'submission_nvarc.json', baseline)
            merge_available()
            event('cpu_flow_complete')
        else:
            if clock()-start >= budget.nvarc_stop: raise TimeoutError('NVARC start deadline')
            # Same setup as exp085, after the all-ID fallback is durable.
            jobs.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'tensorflow'],
                     'tensorflow_uninstall.log', min(budget.nvarc_stop, clock()-start+120))
            if clock()-start >= budget.nvarc_stop: raise TimeoutError('NVARC setup deadline')
            end_wall = time.time() + max(0, start+budget.nvarc_stop-clock())
            jobs.launch([sys.executable, str(payload/'starter.py'), '--end-time', str(end_wall)],
                        'nvarc.log', cwd=payload)
            event('nvarc_started')
            next_save = clock()
            while jobs.p.poll() is None and clock()-start < budget.nvarc_hard-20:
                if clock() >= next_save:
                    try:
                        recovered = collect_bounded(tasks, store, payload, tmp, start+budget.nvarc_stop)
                        submission = recovered; save(out/'submission.json', submission)
                    except Exception as exc: event('recovery_warning', error=repr(exc))
                    next_save = clock()+60
                if clock()-start >= budget.nvarc_stop:
                    event('nvarc_stop_requested'); break
                sleep(1)
            jobs.cleanup()
            event('nvarc_reaped')
            try: submission = collect_bounded(tasks, store, payload, tmp, start+budget.nvarc_hard-20)
            except Exception as exc: event('final_recovery_warning', error=repr(exc))
            baseline = deepcopy(validate(submission, tasks))
            save(out/'submission_nvarc.json', baseline); save(out/'submission.json', baseline)
            counts_path = tmp/'recovered_counts.json'
            try: counts = json.loads(counts_path.read_text())
            except (OSError, ValueError):
                counts = {f'{task}_{idx}': int(row['attempt_1'] != [[0]])
                          for task, rows in baseline.items() for idx, row in enumerate(rows)}
            candidate_counts = counts
            allowed = {r['key']: r for r in select(tasks, None if rerun else 2, counts)}
            from task_trace import summary
            status['nvarc_at_deadline'] = summary(tasks, tmp/'task_trace', start)
            event('qwen_selection', candidate_counts=counts, selected=list(allowed))
            event('nvarc_frozen')
            if clock()-start > budget.nvarc_hard:
                event('qwen_skipped', reason='NVARC cleanup exceeded configured stop + 5m')
            elif not budget.can_start_qwen(clock()-start):
                event('qwen_skipped', reason='less than minimum remaining')
            elif not gpus_free(): event('qwen_skipped', reason='GPU release unverified')
            else:
                model = model_path(explicit=os.environ.get('COMBO_QWEN_MODEL'))
                event('qwen_model', path=model, copied_weights=False)
                from runtime import install
                warnings = []
                setup_limit = min(budget.qwen_stop-60, clock()-start+600)
                py = install(tmp, env, lambda cmd, name: jobs.run(cmd, name, setup_limit), warnings)
                event('runtime_ready', python=py, warnings=warnings)
                # No fallback to the NVARC environment: the pinned runtime is required.
                if py == sys.executable: raise RuntimeError('isolated Qwen runtime unavailable')
                from runtime import preflight
                if not preflight(py, out, lambda cmd, name: jobs.run(cmd, name, setup_limit)):
                    event('qwen_skipped', reason='runtime import self-check failed; see qwen_preflight.log')
                elif not budget.can_start_qwen(clock()-start):
                    event('qwen_skipped', reason='remaining time after runtime setup')
                elif not gpus_free(): event('qwen_skipped', reason='GPU release recheck failed')
                else:
                    save(payload/'ids.json', list(allowed.values()))
                    save(payload/'challenges.json', tasks)
                    run_qwen(jobs, py, payload, out, model, start, budget, rerun,
                             allowed, merge_available, event, clock, sleep)
    except Exception as exc:
        event('stage_failed', error=repr(exc))
    finally:
        jobs.cleanup()
        # Failed/uncompleted requests never touch attempt_2. Retain validated completed results.
        if not (out/'submission_nvarc.json').exists(): save(out/'submission_nvarc.json', baseline)
        try: validate(submission, tasks)
        except Exception:
            submission = deepcopy(baseline); event('rollback_to_nvarc')
        save(out/'submission.json', submission)
        event('final_saved', before_deadline=clock()-start <= budget.final_save)
    return status


def main(payload, out, tmp, tasks, start, cpu=False):
    # Required for wait/reap of workers whose launcher has already died.
    if ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) != 0:
        raise RuntimeError('Linux child subreaper setup failed')
    budget = Budget() if os.environ.get('KAGGLE_IS_COMPETITION_RERUN') else Budget.commit()
    def deadline(sig, frame): raise TimeoutError('final-save safety deadline')
    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, max(.1, start+budget.final_save-30-time.monotonic()))
    try: return orchestrate(tasks, payload, out, tmp, start, cpu)
    finally: signal.setitimer(signal.ITIMER_REAL, 0)
