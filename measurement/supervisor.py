import time
START_MONO = time.monotonic()
START_UTC = time.time()
"""135 minute bounded, sequential B/E/F measurement. MIT-0."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import threading
import zipfile
PAYLOAD_B64 = ''
PAYLOAD_SHA256 = ''
CPU_ONLY = False
ACTIVE = set()
REGISTRY = None
STARTUP_SECONDS = 600
GENERATION_SECONDS = 1800
DRAIN_SECONDS = 30


def update_registry():
    if REGISTRY is not None:
        temp=REGISTRY.with_suffix('.tmp')
        temp.write_text(json.dumps([x.pid for x in ACTIVE]))
        temp.replace(REGISTRY)


def isolated_env(env, payload):
    env=dict(env)
    env['PYTHONPATH']=str(payload)
    env.pop('PYTHONHOME',None)
    env['PYTHONNOUSERSITE']='1'
    env['LD_LIBRARY_PATH']=':'.join(p for p in env.get('LD_LIBRARY_PATH','').split(':')
        if p and not any(s in p for s in ['pip-install-unsloth','site-packages/torch',
                                         'dist-packages/torch']))
    return env


RUNTIME_PROBE = '''
import importlib,json,sys,traceback
from pathlib import Path
result=dict(sys_path=sys.path,ok=True)
for name in ['torch','vllm','vllm._C']:
    try:
        module=importlib.import_module(name)
        result[name]=dict(version=getattr(module,'__version__',None),file=getattr(module,'__file__',None),ok=True)
    except Exception:
        result[name]=dict(ok=False,error=traceback.format_exc())
        if sys.argv[2]!='cpu' or name=='torch':result['ok']=False
print(json.dumps(result),flush=True)
Path(sys.argv[1]).write_text(json.dumps(result,indent=2))
raise SystemExit(0 if result['ok'] else 1)
'''


def stop(p):
    if p is None: return
    try: os.killpg(p.pid, signal.SIGTERM)
    except ProcessLookupError: pass
    try: p.wait(timeout=2)
    except subprocess.TimeoutExpired: pass
    try: os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError: pass
    try: p.wait(timeout=1)
    except subprocess.TimeoutExpired: pass
    ACTIVE.discard(p)
    update_registry()


def guarded(cmd, env, log, deadline, ready=None, startup_seconds=600, cpu=False,
            stop_after_ready=None, condition_deadline=None):
    """All child processes are in their own group, including GPU workers."""
    began=time.monotonic(); stats={}; peaks={}; peak_total=None; reason=None
    log.parent.mkdir(parents=True,exist_ok=True)
    p=subprocess.Popen(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
    ACTIVE.add(p)
    update_registry()
    written=0
    def drain():
        nonlocal written,stats
        with log.open('wb') as f:
            for line in iter(p.stdout.readline,b''):
                if b'SPEED_STATS ' in line:
                    try: stats=json.loads(line.split(b'SPEED_STATS ',1)[1])
                    except ValueError: pass
                block=line[:max(0,32*1024**2-written)]
                f.write(block); written+=len(block)
    t=threading.Thread(target=drain,daemon=True);t.start()
    next_sample=0.; ready_mono=None
    try:
        while p.poll() is None:
            now=time.monotonic()
            if ready is not None and ready.exists() and ready_mono is None:
                ready_mono=json.loads(ready.read_text())['monotonic']
                if stop_after_ready is not None:
                    deadline=min(deadline,ready_mono+stop_after_ready)
            if now>=deadline: reason='global_deadline';break
            if condition_deadline is not None and now>=condition_deadline:
                reason='condition_deadline';break
            if ready_mono is None and now>=began+startup_seconds: reason='startup_timeout';break
            if not cpu and now>=next_sample:
                try:
                    q=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],
                                     capture_output=True,text=True,timeout=1)
                    sample={x.split(',')[0].strip():int(x.split(',')[1]) for x in q.stdout.splitlines()}
                    for k,v in sample.items():peaks[k]=max(peaks.get(k,0),v)
                    if sample:peak_total=max(peak_total or 0,sum(sample.values()))
                except (OSError,ValueError,subprocess.TimeoutExpired): pass
                next_sample=now+1
            time.sleep(.05)
    finally:
        stopped=time.monotonic();stop(p);t.join(timeout=1)
        if not t.is_alive():p.stdout.close()
    return dict(returncode=p.returncode,stop_reason=reason,seconds=stopped-began,
        child_pid=p.pid,ready_monotonic=ready_mono,stop_monotonic=stopped,
        metrics=stats or None,log_written_bytes=written,
        sampled_peak_vram_mib=peaks or None,sampled_peak_total_vram_mib=peak_total,
        log_drain_complete=not t.is_alive())


def mtp_check(model,out):
    """Config + tensor names only. No tensors/weights loaded or copied."""
    from core import save
    config_path=Path(model)/'config.json'; raw=config_path.read_bytes();c=json.loads(raw)
    (out/'mounted_config.json').write_bytes(raw)
    tc=c.get('text_config',c);layers=tc.get('mtp_num_hidden_layers',0)
    index_path=Path(model)/'model.safetensors.index.json'
    index=json.loads(index_path.read_text()) if index_path.exists() else {}
    mapping={k:v for k,v in index.get('weight_map',{}).items() if k.startswith('mtp.')}
    shards=sorted(set(mapping.values()));present=all((Path(model)/n).is_file() for n in shards)
    # Read safetensors JSON headers without materializing any weight tensor.
    header_keys=[]
    for shard in shards:
        if not (Path(model)/shard).exists():continue
        with (Path(model)/shard).open('rb') as f:
            n=int.from_bytes(f.read(8),'little')
            if n>16*1024**2:raise ValueError('unreasonable safetensors header')
            header_keys.extend(json.loads(f.read(n)).keys())
    missing=[k for k in mapping if k not in header_keys]
    supported=c.get('model_type')=='qwen3_5' and layers>0 and bool(mapping) and present and not missing
    r=dict(config_path=str(config_path),config_sha256=hashlib.sha256(raw).hexdigest(),
        model_type=c.get('model_type'),mtp_num_hidden_layers=layers,mtp_index_keys=len(mapping),
        mtp_shards=shards,mtp_shards_present=present,missing_header_keys=missing,
        supported_by_pinned_source=supported,requires_vllm='0.19.1',
        reason='config, indexed weights and v0.19.1 qwen3_5_mtp implementation available' if supported else
        'MTP config or indexed weights missing; B skipped',gpu_execution_verified=False)
    save(out/'mtp_check.json',r);return r


def summarize(folder,row):
    records=[json.loads(p.read_text()) for p in (folder/'records').glob('*.json')]
    through=list(folder.glob('throughput_*.json'))
    t=json.loads(through[0].read_text()) if through else None
    start=folder/'generation_start_0.json'
    if t is None and start.exists():
        elapsed=row['stop_monotonic']-json.loads(start.read_text())['monotonic']
        count=sum(r.get('thinking_tokens',0)+r.get('answer_tokens',0) for r in records)
        checkpoint=folder/'progress.json'
        if checkpoint.exists():count=max(count,json.loads(checkpoint.read_text())['generated_tokens'])
        t=dict(generated_tokens=count,generation_wall_seconds=elapsed,aggregate_tok_s=count/max(elapsed,1e-9),
               partial_lower_bound=True)
    from measurement import counts
    progress=folder/'progress.json'
    row['progress']=json.loads(progress.read_text()) if progress.exists() else None
    if row['progress'] is not None and t and t.get('partial_lower_bound'):
        row['progress']['partial_lower_bound']=True
        row['progress']['observed_until_stop_seconds']=t['generation_wall_seconds']
    row['completion_counts']=counts(records,2 if row.get('actual_request_limit')==2 else 32)
    import re
    log=(folder/'run.log').read_text(errors='replace')
    capacities=re.findall(r'GPU KV cache size: ([0-9,]+) tokens',log)
    row['kv_capacity_tokens']=[int(x.replace(',','')) for x in capacities] or None
    row.update(throughput=t,observed=len(records),completed=sum(r.get('complete',False) for r in records),
        request_errors=[dict(key=r['key'],error=r['error']) for r in records if 'error' in r],
        status='failed' if row['returncode']!=0 else 'partial' if row['stop_reason'] else 'ok')
    if row['stop_reason']: row['status']='partial'
    return row


def conditions(payload,out,py,model,env,deadline,cpu=False,inject=False,stop_after_ready=None,plan=None,force_no_mtp=False):
    from core import save
    env=isolated_env(env,payload)
    rows=[]; fp8_fallback=force_no_mtp
    mtp_supported=cpu or json.loads((out/'mtp_check.json').read_text())['supported_by_pinned_source']
    for name,n in (plan or [('B',16),('E',32),('F',32)]):
        folder=out/name;folder.mkdir(parents=True,exist_ok=True)
        if time.monotonic()>=deadline:
            rows.append(dict(condition=name,status='not_started_global_deadline'));continue
        if name=='B' and not mtp_supported:
            rows.append(dict(condition=name,status='skipped',reason='baseline MTP unavailable; see mtp_check.json'))
            save(out/'conditions.json',rows);continue
        condition_start=time.monotonic()
        condition_deadline=condition_start+STARTUP_SECONDS+GENERATION_SECONDS+DRAIN_SECONDS
        check=guarded([py,'-u','-c',RUNTIME_PROBE,str(folder/'runtime_preflight.json'),
                       'cpu' if cpu else 'gpu'],env,folder/'runtime_preflight.log',min(deadline,condition_start+STARTUP_SECONDS),cpu=True)
        if check['returncode']!=0 or check['stop_reason']:
            rows.append(dict(condition=name,status='skipped',reason='runtime preflight failed',preflight=check))
            save(out/'conditions.json',rows)
            continue
        cmd=[py,'-u',str(payload/'runner.py'),'--payload',str(payload),'--out',str(folder),
             '--model',model,'--deadline',str(deadline-3),'--condition',name,'--max-seqs',str(n),
             '--seconds','60' if cpu else str(GENERATION_SECONDS)]
        if cpu:cmd+=['--cpu','--thinking-cap','2','--answer-cap','2']
        if inject and name=='E':cmd+=['--inject-failure']
        # Test self-stop uses long actual generation, then kills the loaded model.
        if stop_after_ready is not None:cmd+=['--thinking-cap','24576','--answer-cap','4096']
        if not mtp_supported or (name in ['E','F'] and fp8_fallback):cmd+=['--no-mtp']
        attempt_start=condition_start
        r=guarded(cmd,env,folder/'run.log',deadline,folder/'ready.json',cpu=cpu,
                  stop_after_ready=stop_after_ready,condition_deadline=condition_deadline,
                  startup_seconds=max(.01,STARTUP_SECONDS-(time.monotonic()-condition_start)))
        attempts=[]
        if name in ['E','F'] and '--no-mtp' not in cmd and r['ready_monotonic'] is None and r['stop_reason'] in [None,'startup_timeout'] and (r['stop_reason']=='startup_timeout' or r['returncode']!=0) and time.monotonic()<min(deadline,condition_deadline):
            # A timed-out MTP attempt needs a fresh startup window; both attempts
            # remain bounded by the same condition deadline and global deadline.
            attempts.append(dict(mtp_startup=r))
            fp8_fallback=True
            retry=folder/'without_mtp';retry.mkdir(exist_ok=True)
            cmd[cmd.index('--out')+1]=str(retry)
            cmd+=['--no-mtp']
            if '--inject-failure' in cmd:cmd.remove('--inject-failure')
            r=guarded(cmd,env,retry/'run.log',deadline,retry/'ready.json',cpu=cpu,
                startup_seconds=STARTUP_SECONDS if r['stop_reason']=='startup_timeout' else
                    max(.01,STARTUP_SECONDS-(time.monotonic()-attempt_start)),
                stop_after_ready=stop_after_ready,condition_deadline=condition_deadline)
            folder=retry
        if name in ['E','F'] and r['ready_monotonic'] is None:fp8_fallback=True
        r.update(data_dir=str(folder),attempts=attempts,mtp_disabled='--no-mtp' in cmd,
            startup_wall_seconds=(r['ready_monotonic']-attempt_start) if r['ready_monotonic'] else None,condition=name,max_num_seqs=n,actual_request_limit=2 if cpu else n,
                 cpu_mtp_emulated=cpu and name=='E')
        rows.append(summarize(folder,r));save(out/'conditions.json',rows)
        if r['stop_reason']=='global_deadline':deadline=time.monotonic()
    # A late F-only startup failure invalidates the prior E MTP measurement.
    # Preserve it, then measure E without MTP under the unchanged global deadline.
    e=next((r for r in rows if r['condition']=='E'),None)
    f=next((r for r in rows if r['condition']=='F'),None)
    if e and f and not e.get('mtp_disabled',True) and f.get('mtp_disabled'):
        retry_out=out/'paired_fallback';retry_out.mkdir(exist_ok=True)
        if not cpu:save(retry_out/'mtp_check.json',dict(supported_by_pinned_source=True))
        replacement=conditions(payload,retry_out,py,model,env,deadline,cpu,plan=[('E',32)],force_no_mtp=True)[0]
        replacement['superseded_mtp_measurement']=e
        replacement['fallback_note']='F startup failed; E rerun without MTP, subject to original global deadline'
        rows[rows.index(e)]=replacement
    from measurement import compare_answers
    save(out/'answer_comparison.json',compare_answers(out,rows))
    save(out/'conditions.json',rows)
    return rows


def main():
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    out=Path(os.environ.get('QWEN_SPEED_OUT','/kaggle/working/qwen38_s3'));out.mkdir(parents=True,exist_ok=True)
    global REGISTRY
    REGISTRY=out/'watchdog_groups.json';REGISTRY.write_text('[]')
    watchdog_code = """
import json,os,signal,sys,time
from pathlib import Path
parent=int(sys.argv[1]);deadline=float(sys.argv[2]);out=Path(sys.argv[3])
while time.monotonic()<deadline:
    if os.getppid()!=parent:break
    time.sleep(min(.2,max(0,deadline-time.monotonic())))
try:groups=json.loads((out/'watchdog_groups.json').read_text())
except Exception:groups=[]
for pid in groups:
    try:os.killpg(pid,signal.SIGKILL)
    except ProcessLookupError:pass
(out/'hard_stop.json').write_text(json.dumps(dict(source='independent_process',
    seconds=time.monotonic()-(deadline-8095),reason='deadline' if os.getppid()==parent else 'parent_exit')))
try:os.kill(parent,signal.SIGKILL)
except ProcessLookupError:pass
"""
    watchdog=subprocess.Popen([sys.executable,'-c',watchdog_code,str(os.getpid()),
        str(START_MONO+8095),str(out)],start_new_session=True)
    archive=base64.b64decode(PAYLOAD_B64)
    if hashlib.sha256(archive).hexdigest()!=PAYLOAD_SHA256:raise ValueError('bad payload hash')
    payload=out/'payload';payload.mkdir(exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(archive)) as z:z.extractall(payload)
    sys.path.insert(0,str(payload));from core import save
    tmp=Path('/kaggle/temp/sol_s3');tmp.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PIP_NO_INDEX='1',PIP_NO_CACHE_DIR='1',
        TMPDIR=str(tmp),XDG_CACHE_HOME=str(tmp/'cache'),HF_HOME=str(tmp/'hf'),
        TORCHINDUCTOR_CACHE_DIR=str(tmp/'inductor'),TRITON_CACHE_DIR=str(tmp/'triton'),
        VLLM_CACHE_ROOT=str(tmp/'vllm'),CUDA_CACHE_PATH=str(tmp/'cuda'),PYTHONDONTWRITEBYTECODE='1',
        TOKENIZERS_PARALLELISM='false',VLLM_USE_DEEP_GEMM='0',VLLM_NO_USAGE_STATS='1',
        VLLM_WORKER_MULTIPROC_METHOD='spawn',CUDA_VISIBLE_DEVICES='' if CPU_ONLY else '0,1,2,3')
    env=isolated_env(env,payload)
    deadline=START_MONO+8070;warnings=[];result=dict(cpu_only=CPU_ONLY,start_utc=START_UTC)
    try:
        models=[]
        for path in Path('/kaggle/input').rglob('config.json'):
            try:
                c=json.loads(path.read_text())
                if c.get('model_type')==('qwen3' if CPU_ONLY else 'qwen3_5'):models.append(str(path.parent))
            except (ValueError,OSError):pass
        if CPU_ONLY:
            from cpu_fixture import build_tiny
            models=[str(build_tiny(tmp/'tiny'))]
        if len(models)!=1:raise RuntimeError(f'expected exactly one model mount, found {models}')
        model=models[0];save(out/'mount.json',dict(selected=model,copied_weights=False))
        if CPU_ONLY:
            # Inspect the actual 27B mount without loading any weight tensors.
            large=[]
            for path in Path('/kaggle/input').rglob('config.json'):
                try:
                    if json.loads(path.read_text()).get('model_type')=='qwen3_5':large.append(str(path.parent))
                except (OSError,ValueError):pass
            if len(large)==1:
                try:result['mtp_mount_inspection']=mtp_check(large[0],out)
                except Exception as exc:result['mtp_mount_inspection']=dict(error=repr(exc))
            else:result['mtp_mount_inspection']=dict(error='expected one 27B mount',found=large)
            result['cycles']=[]
            for cycle in range(3):
                folder=out/f'cycle{cycle}';folder.mkdir(exist_ok=True)
                rows=conditions(payload,folder,sys.executable,model,env,deadline,True,inject=True)
                timeout_rows=conditions(payload,folder/'self_stop',sys.executable,model,env,deadline,True,stop_after_ready=.1)
                ok=([r['condition'] for r in rows]==['B','E','F'] and
                    rows[0]['completed']==2 and rows[1]['completed']==2 and bool(rows[1]['attempts']) and rows[2]['completed']==2 and
                    timeout_rows[0]['stop_reason']=='global_deadline' and timeout_rows[0]['ready_monotonic'] is not None and
                    timeout_rows[0]['seconds']<600 and
                    all(r['status']=='not_started_global_deadline' for r in timeout_rows[1:]))
                result['cycles'].append(dict(cycle=cycle,conditions=rows,self_stop=timeout_rows,passed=ok))
                save(out/'result.json',result)
            result['cpu_flow_validation_passed']=all(c['passed'] for c in result['cycles'])
        else:
            try:result['mtp']=mtp_check(model,out)
            except Exception as exc:
                result['mtp']=dict(supported_by_pinned_source=False,reason=repr(exc))
                save(out/'mtp_check.json',result['mtp'])
            from runtime import install
            def run(cmd,name):
                return guarded(cmd[:1]+['-I']+cmd[1:],env,out/name,min(deadline,START_MONO+360),startup_seconds=360,cpu=True)['returncode']
            py=install(tmp,env,run,warnings)
            result['conditions']=conditions(payload,out,py,model,env,deadline)
    except BaseException as exc:result['error']=repr(exc)
    finally:
        for p in list(ACTIVE):stop(p)
        result.update(seconds=time.monotonic()-START_MONO,warnings=warnings,
            deadline_seconds=8070,hard_stop_seconds=8095,wall_limit_seconds=8100,
            quota_hours_upper_bound=8100/3600*2,
            memory_note='nvidia-smi 1-second samples; transient peaks can be missed',
            distribution_note='acceptance rate is diagnostic; it does not establish equality of answer distributions')
        save(out/'result.json',result)
        print(json.dumps(result),flush=True)
        watchdog.terminate();watchdog.wait(timeout=1)
    return 0

if __name__=='__main__':raise SystemExit(main())
