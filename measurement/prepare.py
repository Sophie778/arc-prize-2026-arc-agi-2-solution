"""Portable S3 workload selection / private payload builder (MIT-0).

Nothing runs on import. Inputs and outputs contain competition data: keep them
outside the public measurement directory and never publish generated kernels.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import random
import zipfile

HERE = Path(__file__).resolve().parent
CODE = ['core.py', 'runner.py', 'tiny_engine.py', 'runtime.py', 'telemetry.py',
        'measurement.py', 'cpu_fixture.py', 'system_prompt.txt', 'chat_template.jinja']


def select(outputs, baseline, tasks):
    """Match the historical selection, given the same private inputs.

    outputs: frozen diagnostic list of {task, idx, key}; baseline: key -> top-2
    grids in ranking order; tasks: public evaluation task JSON including labels.
    Neither labels nor predictions are included in the measurement payload.
    """
    rows = [dict(r) for r in outputs]
    if len({r['key'] for r in rows}) != len(rows):
        raise ValueError('duplicate output key')
    def attempts(r):
        return baseline.get(r['key'], [])[:2]
    def hit(r):
        return tasks[r['task']]['test'][r['idx']]['output'] in attempts(r)
    rng = random.Random(20261022)
    misses = sorted([r for r in rows if attempts(r) and not hit(r)], key=lambda r: r['key'])
    hits = sorted([r for r in rows if hit(r)], key=lambda r: r['key'])
    chosen = rng.sample(misses, 12) + rng.sample(hits, 4)
    rng.shuffle(chosen)
    return chosen


def build(outputs, baseline, tasks, destination):
    destination = Path(destination).resolve()
    if destination == HERE or HERE in destination.parents:
        raise ValueError('private output must be outside the public measurement directory')
    chosen = select(outputs, baseline, tasks)
    challenges = {r['task']: dict(train=tasks[r['task']]['train'],
        test=[dict(input=p['input']) for p in tasks[r['task']]['test']]) for r in chosen}
    from core import prompt
    template = (HERE / 'system_prompt.txt').read_text()
    cpu = []
    for r in sorted(chosen, key=lambda r: (len(prompt(challenges[r['task']], r['idx'], template)), r['key'])):
        if r['task'] not in {x['task'] for x in cpu}:
            cpu.append(r)
        if len(cpu) == 2:
            break
    files = {n: (HERE / n).read_bytes() for n in CODE}
    for name, value in [('ids.json', chosen), ('baseline_ids.json', chosen),
                        ('cpu_ids.json', cpu), ('challenges.json', challenges)]:
        files[name] = json.dumps(value).encode()
    manifest = {n: hashlib.sha256(v).hexdigest() for n, v in files.items()}
    files['manifest.json'] = json.dumps(manifest).encode()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, (2026, 10, 24, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    payload = buf.getvalue()
    source = (HERE / 'supervisor.py').read_text()
    source = source.replace("PAYLOAD_B64 = ''", 'PAYLOAD_B64 = ' + repr(base64.b64encode(payload).decode()))
    source = source.replace("PAYLOAD_SHA256 = ''", 'PAYLOAD_SHA256 = ' + repr(hashlib.sha256(payload).hexdigest()))
    for cpu_only, folder in [(False, 'kernel'), (True, 'kernel_cpu')]:
        target = destination / folder
        target.mkdir(parents=True, exist_ok=True)
        (target / 'kernel.py').write_text(source.replace('CPU_ONLY = False', 'CPU_ONLY = ' + str(cpu_only)))
        meta = json.loads((HERE / 'kernel/kernel-metadata.json').read_text())
        if cpu_only:
            meta.update(id=meta['id'] + '-cpu', title=meta['title'] + '-cpu', enable_gpu=False)
            meta.pop('machine_shape', None)
        (target / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    # Private sidecar, not part of the publication bundle.
    (destination / 'payload_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Prepared private kernels; generated files contain task data and must not be published.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outputs', type=Path, required=True, help='frozen diagnostic outputs JSON: list or {outputs: list}')
    p.add_argument('--baseline', type=Path, required=True, help='JSON mapping output key to ranked top-2 grids')
    p.add_argument('--evaluation', type=Path, required=True, help='directory of public evaluation task JSON files')
    p.add_argument('--output', type=Path, required=True, help='private directory outside this publication bundle')
    a = p.parse_args()
    outputs = json.loads(a.outputs.read_text())
    if isinstance(outputs, dict):
        outputs = outputs['outputs']
    baseline = json.loads(a.baseline.read_text())
    tasks = {r['task']: json.loads((a.evaluation / (r['task'] + '.json')).read_text()) for r in outputs}
    build(outputs, baseline, tasks, a.output)


if __name__ == '__main__':
    main()
