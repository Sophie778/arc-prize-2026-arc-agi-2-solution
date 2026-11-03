"""Record interpreter provenance and verify the measured Qwen binary pair. MIT-0."""
import importlib.metadata as md
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import traceback
import zipfile


def main():
    mode, dest, wheel_dir = sys.argv[1:]
    report = dict(mode=mode, python=sys.version, executable=sys.executable,
        prefix=sys.prefix, base_prefix=sys.base_prefix, libc=platform.libc_ver(),
        sys_path=sys.path, env={k: os.environ.get(k) for k in
            ['PYTHONPATH', 'PYTHONHOME', 'LD_LIBRARY_PATH']},
        pth={str(p): p.read_text(errors='replace') for base in sys.path if Path(base).is_dir()
             for p in Path(base).glob('*.pth')},
        pyvenv_cfg=(Path(sys.prefix)/'pyvenv.cfg').read_text()
            if (Path(sys.prefix)/'pyvenv.cfg').exists() else None)
    for name in ['torch', 'vllm']:
        try:
            spec = importlib.util.find_spec(name)
            report[name+'_origin'] = spec.origin if spec else None
            report[name+'_version'] = md.version(name)
            report[name+'_requires_torch'] = [r for r in md.requires(name) or []
                                              if r.lower().startswith('torch')]
        except Exception as exc: report[name+'_metadata_error'] = repr(exc)
    report['wheel_torch_requirements'] = {}
    for wheel in (Path('/kaggle/input').rglob('vllm-*.whl') if wheel_dir == '/nonexistent'
                  else Path(wheel_dir).glob('vllm-*.whl')):
        with zipfile.ZipFile(wheel) as z:
            meta = next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))
            report['wheel_torch_requirements'][wheel.name] = [line for line in
                z.read(meta).decode().splitlines() if line.startswith('Requires-Dist: torch')]
    rc = 0
    try:
        import torch
        report.update(torch_import_version=torch.__version__, torch_import_file=torch.__file__,
                      torch_cuda=torch.version.cuda)
        if mode != 'audit':
            assert torch.__version__ == '2.10.0+cu128', 'unexpected torch build'
            assert Path(torch.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), 'external torch'
            assert md.version('vllm') == '0.19.1', 'unexpected vLLM version'
            assert Path(report['vllm_origin']).resolve().is_relative_to(Path(sys.prefix).resolve()), 'external vLLM'
            import vllm
            report['vllm_import_file'] = vllm.__file__
            report['vllm_import_ok'] = True
            try:
                import vllm._C
                report['extension_import_ok'] = True
            except Exception:
                report['extension_import_ok'] = False
                report['extension_error'] = traceback.format_exc()
                if mode != 'cpu': raise
                report['cpu_limitation'] = 'CPU probe cannot certify CUDA extension compatibility'
        report['loaded_torch_cuda_libraries'] = [l for l in Path('/proc/self/maps').read_text().splitlines()
                                                if any(s in l for s in ['libtorch', 'libc10', 'libcuda'])]
    except Exception:
        report['error'] = traceback.format_exc()
        rc = 1
    report['ok'] = rc == 0
    Path(dest).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
    return rc


if __name__ == '__main__': sys.exit(main())
