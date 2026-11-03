"""Log cumulative scheduler counters; no correctness scoring. MIT-0."""
import json
from vllm.v1.metrics.loggers import StatLoggerBase

class SpeedLogger(StatLoggerBase):
    def __init__(self, vllm_config, engine_index=0):
        self.preemptions=0; self.drafted=0; self.accepted=0; self.drafts=0
        self.peak_kv=0.; self.records=0
    def record(self, scheduler_stats, iteration_stats, mm_cache_stats=None, engine_idx=0):
        self.records+=1
        if iteration_stats is not None:
            self.preemptions+=iteration_stats.num_preempted_reqs
        if scheduler_stats is not None:
            self.peak_kv=max(self.peak_kv,scheduler_stats.kv_cache_usage)
            s=scheduler_stats.spec_decoding_stats
            if s is not None:
                self.drafted+=s.num_draft_tokens; self.accepted+=s.num_accepted_tokens; self.drafts+=s.num_drafts
        self.log()
    def log_engine_initialized(self): self.log()
    def log(self):
        print('SPEED_STATS '+json.dumps(dict(preemptions=self.preemptions,drafted=self.drafted,
            accepted=self.accepted,drafts=self.drafts,peak_kv_usage=self.peak_kv,records=self.records,
            acceptance_rate=self.accepted/self.drafted if self.drafted else None)),flush=True)
