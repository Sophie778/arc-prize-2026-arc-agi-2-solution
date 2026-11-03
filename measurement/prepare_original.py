"""Package frozen v2 prompts and private CPU/GPU artifacts. MIT-0."""
from pathlib import Path
import base64,hashlib,io,json,zipfile
H=Path(__file__).resolve().parent

def sha(b):return hashlib.sha256(b).hexdigest()
def build():
    names=['core.py','runner.py','tiny_engine.py','runtime.py','telemetry.py','measurement.py','cpu_fixture.py','system_prompt.txt',
           'chat_template.jinja','challenges.json','ids.json','cpu_ids.json','baseline_ids.json']
    files={n:(H/n).read_bytes() for n in names}
    manifest={n:sha(v) for n,v in files.items()};files['manifest.json']=json.dumps(manifest).encode()
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
        for n,v in sorted(files.items()):
            i=zipfile.ZipInfo(n,(2026,10,24,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;z.writestr(i,v)
    source=(H/'supervisor.py').read_text().replace("PAYLOAD_B64 = ''",'PAYLOAD_B64 = '+repr(base64.b64encode(b.getvalue()).decode()))
    source=source.replace("PAYLOAD_SHA256 = ''",'PAYLOAD_SHA256 = '+repr(sha(b.getvalue())))
    result={}
    for cpu,folder in [(False,'kernel'),(True,'kernel_cpu')]:
        target=H/folder/'kernel.py';target.write_text(source.replace('CPU_ONLY = False','CPU_ONLY = '+str(cpu)))
        meta=json.loads((H.parent/'qwen38_fast'/folder/'kernel-metadata.json').read_text())
        suffix='-cpu' if cpu else ''
        meta.update(id='sophie555/arc26-qwen38-s3'+suffix,title='arc26-qwen38-s3'+suffix)
        if cpu:
            meta['model_sources']=[]
            meta['model_sources'].append('mikedan7/qwen3-8-27b-fp8-official/pytorch/hf-fp8/1')
        (H/folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
        result[folder]=dict(sha256=sha(target.read_bytes()),metadata_sha256=sha((H/folder/'kernel-metadata.json').read_bytes()))
    (H/'payload_manifest.json').write_text(json.dumps(dict(payload=manifest,artifacts=result),indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':build()
