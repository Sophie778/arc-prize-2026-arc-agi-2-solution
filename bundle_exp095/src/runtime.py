"""Install the measured cp311 runtime without NVARC path inheritance. MIT-0."""
import json
from pathlib import Path
import sys

def install(tmp, env, run, warnings):
    manifests=list(Path('/kaggle/input').rglob('runtime_manifest.json'))
    selected=None
    for path in manifests:
        try:
            if 'vllm==0.19.1' in json.loads(path.read_text()).get('requirements',[]):selected=path;break
        except (OSError,ValueError):pass
    if selected is None:
        warnings.append('cp311 wheels not found; attempt image runtime');return sys.executable
    # Jobs holds this dictionary by reference. Change it only after NVARC is reaped.
    env['PYTHONPATH'] = str(Path(__file__).resolve().parent)
    env.pop('PYTHONHOME', None)
    env['PYTHONNOUSERSITE'] = '1'
    env['LD_LIBRARY_PATH'] = ':'.join(p for p in env.get('LD_LIBRARY_PATH', '').split(':')
        if p and not any(s in p for s in ['pip-install-unsloth', 'site-packages/torch',
                                         'dist-packages/torch']))
    target=tmp/'venv' 
    if run([sys.executable,'-I','-m','venv','--without-pip',str(target)],'venv.log') != 0:
        warnings.append('venv creation failed; attempt image runtime');return sys.executable
    py=str(target/'bin/python'); wheels=selected.parent/'wheels'
    pip=next(wheels.glob('pip-*.whl'),None)
    if pip is None:
        warnings.append('bootstrap pip wheel absent; attempt image runtime');return sys.executable
    bootstrap=f'import sys,runpy;sys.path.insert(0,{str(pip)!r});runpy.run_module("pip",run_name="__main__")'
    rc=run([py,'-I','-c',bootstrap,'install','--no-index','--find-links',str(wheels),'--require-hashes',
            '--no-cache-dir','-r',str(selected.parent/'requirements.lock')],'offline_install.log')
    if rc != 0:
        warnings.append('offline install incomplete; attempt image runtime');return sys.executable
    if run([py,'-I','-m','pip','check'],'pip_check.log') != 0:warnings.append('pip check mismatch; continuing')
    return py


def preflight(py, out, run, cpu=False):
    probe = Path(__file__).resolve().with_name('runtime_probe.py')
    return run([py, '-I', str(probe), 'cpu' if cpu else 'check',
                str(out/'qwen_preflight.json'), '/nonexistent'], 'qwen_preflight.log') == 0
