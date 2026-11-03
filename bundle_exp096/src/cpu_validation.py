"""Real full ARC prompts, exact shared runner, synthetic tiny CPU model, 3 loops. MIT-0."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from core import save
from policy import select


def fixture(root):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    vocab={'[UNK]':0,'[PAD]':1,'</think>':2,'[EOS]':3}
    vocab.update({f'word{i}':i for i in range(4,32)})
    tokenizer=Tokenizer(models.WordLevel(vocab,unk_token='[UNK]'))
    tokenizer.pre_tokenizer=pre_tokenizers.WhitespaceSplit()
    tok=PreTrainedTokenizerFast(tokenizer_object=tokenizer,unk_token='[UNK]',pad_token='[PAD]',
        eos_token='[EOS]',additional_special_tokens=['</think>'],model_max_length=65536)
    root.mkdir(exist_ok=True);tok.save_pretrained(root)
    torch.manual_seed(0)
    config=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=1,head_dim=8,max_position_embeddings=65536,
        eos_token_id=3,pad_token_id=1)
    net=Qwen3ForCausalLM(config);net.save_pretrained(root)
    return sum(p.numel() for p in net.parameters())


def run(payload, out, tasks):
    out.mkdir(parents=True,exist_ok=True)
    from nvarc_cpu_validation import run as validate_nvarc
    from qwen_config import settings
    rectangular = {k:t for k,t in tasks.items() if any(len(x['input']) != len(x['input'][0]) for x in t['test'])}
    assert len(rectangular) >= 3, 'CPU validation requires at least three real rectangular tasks'
    keys=sorted(rectangular,key=lambda k:(len(json.dumps(rectangular[k])),k))[:3]
    tasks={k:tasks[k] for k in keys}
    views=validate_nvarc(tasks, cycles=3)
    save(out/'views.json',dict(generation_n=1,views_per_output=8,qwen=settings(),**views))
    print('CPU validated generation n=1 views=8; Qwen '+json.dumps(settings()),flush=True)
    parameters=fixture(out/'tiny')
    keys=sorted(tasks,key=lambda k:(len(json.dumps(tasks[k])),k))[:2]
    tasks={k:tasks[k] for k in keys};rows=select(tasks,2)
    save(payload/'challenges.json',tasks);save(payload/'ids.json',rows);save(payload/'cpu_ids.json',rows)
    with (out/'runner.log').open('wb') as log:
        subprocess.run([sys.executable,str(payload/'runner.py'),'--payload',str(payload),
            '--out',str(out/'generation'),'--model',str(out/'tiny'),
            '--deadline',str(time.monotonic()+300),'--cpu','--thinking-cap','8','--answer-cap','8'],
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),
            stdout=log,stderr=subprocess.STDOUT,check=True,timeout=330)
    records=[json.loads(p.read_text()) for p in (out/'generation/records').glob('*.json')]
    report=dict(parameters=parameters,cycles=sorted({r['cycle'] for r in records}),
        records=len(records),all_complete=all(r['complete'] for r in records),
        errors=[r.get('error') for r in records if r.get('error')],
        effort=sorted({r['effort'] for r in records}),inputs=[r['key'] for r in rows],
        limitation='Synthetic one-layer CPU model; no GPU speed, quality, MTP or KV capacity evidence.')
    save(out/'result.json',report)
    assert report['cycles']==[0,1,2] and len(records)==6 and report['all_complete'] and not report['errors']
    assert report['effort']==['xhigh']
    return report


if __name__=='__main__':
    payload,out,tasks_path=map(Path,sys.argv[1:])
    print(json.dumps(run(payload,out,json.loads(tasks_path.read_text()))))
