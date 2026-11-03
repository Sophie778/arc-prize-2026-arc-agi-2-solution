"""Direct-grid prompt, conservative extraction and output/task scoring (MIT-0)."""
import hashlib
import json
from pathlib import Path
import re
WRITTEN_BYTES = 0


def save(path, value):
    global WRITTEN_BYTES
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    data=json.dumps(value,ensure_ascii=False).encode()
    tmp.write_bytes(data);tmp.replace(path)
    WRITTEN_BYTES += len(data)


def prompt(task, idx, template):
    examples=''
    for i,pair in enumerate(task['train']):
        examples+=f'--Example {i}-- \n\n INPUT: \n\n'+json.dumps(pair['input'])+'\n\n'
        examples+='OUTPUT: \n\n'+json.dumps(pair['output'])+'\n\n'
    return template.format(training_examples=examples,test_input=json.dumps(task['test'][idx]['input']))


def valid_grid(g):
    return (isinstance(g,list) and 1<=len(g)<=30 and isinstance(g[0],list)
            and 1<=len(g[0])<=30 and all(isinstance(r,list) and len(r)==len(g[0])
            and all(type(v) is int and 0<=v<=9 for v in r) for r in g))


def extract(text):
    # Only the answer stage is eligible. Never mine a truncated thought for grids.
    text=text.rsplit('</think>',1)[-1]
    if '<think>' in text:return None
    decoder=json.JSONDecoder();grids=[];starts=[];depth=0
    for i,c in enumerate(text):
        if c=='[':
            if depth==0:starts.append(i)
            depth+=1
        elif c==']':depth=max(0,depth-1)
    for start in starts:
        try:
            g,_=decoder.raw_decode(text[start:])
            if valid_grid(g):grids.append(g)
        except ValueError:pass
    if grids:
        # Multiple different final grids are ambiguous, not a best-of-N answer.
        return grids[0] if all(g==grids[0] for g in grids) else None
    stripped=text.strip().strip('`').strip()
    if stripped.startswith('json\n'):stripped=stripped[5:]
    rows=stripped.splitlines()
    if rows and all(re.fullmatch(r'\s*[0-9](?:[ ,]+[0-9])*\s*',r) for r in rows):
        g=[[int(v) for v in re.findall(r'[0-9]',r)] for r in rows]
        return g if valid_grid(g) else None
    return None

