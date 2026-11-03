import time
START = time.monotonic()
"""exp095 standalone bootstrap. Fallback precedes runtime and model imports. MIT-0."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import sys
import zipfile

PAYLOAD_B64 = ''
PAYLOAD_SHA256 = ''
CPU_ONLY = False


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    out = Path(os.environ.get('COMBO_OUT', '/kaggle/working'))
    out.mkdir(parents=True, exist_ok=True)
    comp = next((p for p in [Path('/kaggle/input/competitions/arc-prize-2026-arc-agi-2'),
        Path('/kaggle/input/arc-prize-2026-arc-agi-2')] if p.is_dir()), None)
    if os.environ.get('COMBO_COMP_DIR'): comp = Path(os.environ['COMBO_COMP_DIR'])
    if comp is None: raise RuntimeError('competition mount not found')
    rerun = bool(os.environ.get('KAGGLE_IS_COMPETITION_RERUN'))
    tasks = json.loads((comp/('arc-agi_test_challenges.json' if rerun else
                              'arc-agi_evaluation_challenges.json')).read_text())
    if CPU_ONLY:
        rectangular = [k for k,t in tasks.items() if any(len(x['input']) != len(x['input'][0]) for x in t['test'])]
        keys = sorted(rectangular, key=lambda k: (len(json.dumps(tasks[k])), k))[:3]
        tasks = {k: tasks[k] for k in keys}
    # All IDs/test indices exist from the start, even if payload/runtime setup fails.
    fallback = {k: [dict(attempt_1=[[0]], attempt_2=[[0]]) for _ in v['test']]
                for k, v in tasks.items()}
    target = out/'submission.json'
    tmp_json = out/'submission.json.tmp'; tmp_json.write_text(json.dumps(fallback)); tmp_json.replace(target)
    tmp = Path(os.environ.get('COMBO_TEMP', '/kaggle/temp/exp095'))
    tmp.mkdir(parents=True, exist_ok=True)
    payload = tmp/'payload'; payload.mkdir(exist_ok=True)
    archive = base64.b64decode(PAYLOAD_B64)
    if hashlib.sha256(archive).hexdigest() != PAYLOAD_SHA256: raise ValueError('payload hash mismatch')
    with zipfile.ZipFile(io.BytesIO(archive)) as z: z.extractall(payload)
    manifest = json.loads((payload/'manifest.json').read_text())
    for name, digest in manifest.items():
        if hashlib.sha256((payload/name).read_bytes()).hexdigest() != digest:
            raise ValueError('payload member hash mismatch: ' + name)
    (tmp/'challenges.json').write_text(json.dumps(tasks))
    sys.path.insert(0, str(payload))
    from supervisor import main as supervise
    result = supervise(payload, out, tmp, tasks, START, CPU_ONLY)
    if CPU_ONLY:
        # Same Docker and both runtime sources, without loading model weights.
        from supervisor import Jobs
        from runtime import install, preflight
        inherited = dict(os.environ)
        qenv = dict(inherited, PYTHONDONTWRITEBYTECODE='1', PIP_NO_INDEX='1',
                    PIP_NO_CACHE_DIR='1', TMPDIR=str(tmp))
        jobs = Jobs(qenv, tmp, out, time.monotonic())
        warnings = []
        try:
            py = install(tmp, qenv, lambda cmd, name: jobs.run(cmd, name, 1800), warnings)
            if py == sys.executable: raise RuntimeError('CPU pinned runtime install failed')
            clean = dict(qenv)
            # Reproduce the old interpreter search path after a successful install.
            qenv.clear(); qenv.update(inherited)
            jobs.run([py, str(payload/'runtime_probe.py'), 'audit',
                      str(out/'runtime_inherited.json'), '/nonexistent'],
                     'runtime_inherited.log', 1800)
            qenv.clear(); qenv.update(clean)
            ok = preflight(py, out, lambda cmd, name: jobs.run(cmd, name, 1800), cpu=True)
            (out/'runtime_cpu_result.json').write_text(json.dumps(dict(ok=ok, warnings=warnings)))
            if not ok: raise RuntimeError('CPU runtime preflight failed')
            rc = jobs.run([py, str(payload/'cpu_validation.py'), str(payload),
                           str(out/'cpu_validation'), str(tmp/'challenges.json')],
                          'cpu_validation.log', 1800)
            if rc != 0: raise RuntimeError('CPU real-input three-loop validation failed')
        finally: jobs.cleanup()
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
