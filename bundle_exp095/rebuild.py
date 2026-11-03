"""Rebuild exact submitted kernel bytes, offline; never executes the solver."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

H = Path(__file__).resolve().parent

def digest(b): return hashlib.sha256(b).hexdigest()

def build(verify_only=False):
    identity = json.loads((H/'provenance.json').read_text())
    manifest = json.loads((H/'src/manifest.json').read_text())
    assert set(manifest) | {'manifest.json'} == set(identity['members'])
    files = {n: (H/'src'/n).read_bytes() for n in identity['members']}
    for n, b in files.items():
        assert digest(b) == identity['members'][n]['sha256'], n
        if n in manifest: assert digest(b) == manifest[n], n
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for n,b in sorted(files.items()):
            info=zipfile.ZipInfo(n,(2026,10,24,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,b)
    payload=buf.getvalue()
    assert digest(payload)==identity['payload_sha256']
    text=(H/'entry.py').read_text().replace("PAYLOAD_B64 = ''",
        'PAYLOAD_B64 = '+repr(base64.b64encode(payload).decode())).replace(
        "PAYLOAD_SHA256 = ''", 'PAYLOAD_SHA256 = '+repr(digest(payload)))
    result=text.encode()
    assert digest(result)==identity['kernel_sha256']
    metadata=(H/'notebook/kernel-metadata.json').read_bytes()
    assert digest(metadata)==identity['metadata_sha256']
    if not verify_only:
        target=H/'rebuilt';target.mkdir(exist_ok=True)
        (target/'kernel.py').write_bytes(result)
        (target/'kernel-metadata.json').write_bytes(metadata)
    print(json.dumps(dict(kernel_sha256=digest(result),payload_sha256=digest(payload))))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify-only', action='store_true')
    build(parser.parse_args().verify_only)
