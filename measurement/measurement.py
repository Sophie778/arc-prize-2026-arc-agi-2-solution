"""Wall-time token bins and paired natural-answer diagnostics (MIT-0)."""
import json
import time
from core import save, extract

class Progress:
    def __init__(self, out, start, width=300):
        self.out=out; self.start=start; self.width=width; self.total=0
        self.bins={}; self.last_save=start
    def add(self, count):
        now=time.monotonic(); i=int((now-self.start)//self.width)
        self.bins[i]=self.bins.get(i,0)+count; self.total+=count
        if now-self.last_save>=1:
            self.write(now); self.last_save=now
    def write(self, now):
        elapsed=max(now-self.start,1e-9)
        rows=[]
        for i in range(int(elapsed//self.width)+1):
            duration=min(self.width,max(0,elapsed-i*self.width))
            if duration<=1e-8:continue
            count=self.bins.get(i,0)
            rows.append(dict(start_seconds=i*self.width,end_seconds=i*self.width+duration,
                generated_tokens=count,tok_s=count/duration,complete_window=duration==self.width))
        save(self.out/'progress.json',dict(generated_tokens=self.total,generation_wall_seconds=elapsed,
            aggregate_tok_s=self.total/elapsed,windows=rows,binning='token arrival monotonic time'))
    def finish(self): self.write(time.monotonic())

def records(out):
    return [json.loads(p.read_text()) for p in (out/'records').glob('*.json')]

def counts(items, expected):
    return dict(natural=sum(bool(r.get('natural_end')) for r in items),
        cap_reached=sum(bool(r.get('thinking_capped') or r.get('answer_finish_reason')=='length') for r in items),
        complete=sum(bool(r.get('complete')) for r in items),
        unfinished=expected-sum(bool(r.get('complete')) for r in items),
        note='cap_reached includes incomplete answers; natural requires both stages to stop naturally')

def compare_answers(out, rows):
    base={(r['key'],r['seed']):r for r in records(out/'B')}
    result=[]
    for row in rows:
        name=row['condition']
        if name=='B':continue
        folder=__import__('pathlib').Path(row['data_dir']) if row.get('data_dir') else out/name
        for r in records(folder):
            b=base.get((r['key'],r['seed']))
            eligible=bool(b and b.get('natural_end') and r.get('natural_end'))
            bg=extract(b['answer_text']) if eligible else None
            rg=extract(r['answer_text']) if eligible else None
            result.append(dict(condition=name,key=r['key'],seed=r['seed'],both_natural=eligible,
                both_grids_extractable=eligible and bg is not None and rg is not None,
                grid_equal=(bg==rg) if eligible and bg is not None and rg is not None else None,
                answer_text_equal=(b['answer_text']==r['answer_text']) if eligible else None))
    return dict(pairs=result,note='paired by key and seed; no ground truth, no equivalence claim')
