"""Real 0.6B CPU backend for the shared production scheduler. MIT-0."""
import asyncio
from types import SimpleNamespace
import torch
from transformers import AutoModelForCausalLM

class Params:
    def __init__(self, **kw): self.__dict__.update(kw)

class TinyEngine:
    def __init__(self, model, tokenizer):
        torch.set_num_threads(4)
        self.model = AutoModelForCausalLM.from_pretrained(model, local_files_only=True,
                         torch_dtype=torch.float32).eval()
        self.tok = tokenizer
        self.eos = self.model.generation_config.eos_token_id
        self.eos = self.eos if isinstance(self.eos, list) else [self.eos]
        self.aborted = set()

    async def generate(self, inp, params, request):
        generated = []; cache = None
        rng=torch.Generator(device='cpu').manual_seed(params.seed)
        current = torch.tensor([inp['prompt_token_ids']], dtype=torch.long)
        def step(current, cache):
            with torch.inference_mode():
                output = self.model(input_ids=current, past_key_values=cache, use_cache=True)
                values, indices = torch.topk(output.logits[0, -1].float()/params.temperature, params.top_k)
                probs = torch.softmax(values, dim=-1)
                keep = probs.cumsum(-1)-probs < params.top_p
                probs = probs*keep; probs /= probs.sum()
                return indices[torch.multinomial(probs, 1, generator=rng)].item(), output.past_key_values
        for i in range(params.max_tokens):
            if request in self.aborted: return
            token, cache = await asyncio.to_thread(step, current, cache)
            generated.append(token); current = torch.tensor([[token]], dtype=torch.long)
            stop = token in self.eos or token in (params.stop_token_ids or [])
            finished = stop or i+1 == params.max_tokens
            completion = SimpleNamespace(token_ids=generated[:], text=self.tok.decode(generated, skip_special_tokens=False),
                finish_reason=('stop' if stop else 'length') if finished else None,
                stop_reason=token if stop else None)
            yield SimpleNamespace(outputs=[completion], finished=finished)
            if finished: return
    async def abort(self, request): self.aborted.add(request)
    def shutdown(self): pass
