# Figure 2: Qwen3.8-27B sustained measurement (S3)

This is the data-free reproduction package for the paper's main measurement,
not either final submission. It supplies the measurement implementation, settings,
workload-selection algorithm and anonymous metrics. No competition tasks, task or
output identifiers, labels, predictions, generated answers, thought text, token-ID
sequences or model weights are distributed. Run All in the companion notebook
only exports/verifies these files; it does not rerun the GPU measurement.

## Measurement and completion definition

Historical run: S3 v1, four NVIDIA L4 GPUs, Qwen3.8-27B official FP8 weights,
TP4, vLLM 0.19.1, low reasoning, 24,576 thought tokens, 4,096 answer tokens,
temperature 1, top_p .95, top_k 20. There are 16 selected outputs, each submitted
with seed 0 and seed 1 in seed-major order, hence 32 requests. Each arm starts
with a fresh engine. Both arms ran once, with a 1,800-second generation window;
startup is excluded from throughput. See `runner.py`, `supervisor.py` and
`README_original.md` for exact scheduler, fallback and watchdog behavior.

| Arm | KV cache | MTP draft tokens | Concurrency | Aggregate tok/s | Completed / 32 |
|---|---|---:|---:|---:|---:|
| B (Figure 2) | auto/BF16 | 3 | 16 | 236.260181 | 11 |
| E (Figure 2) | FP8 | 3 | 32 | 418.370938 | 0 |
| F (incomplete additional arm) | FP8 | 4 | 32 | 155.701123 | 0 |

F generated for only 212.317 seconds, with engine errors recorded locally. It is
not a comparable 30-minute arm and is not used in Figure 2. Historical README
specifies planned controls; saved metrics describe what actually ran. A previous
ledger interpretation attributed F's short duration to the global deadline;
this release does not infer a cause from the anonymous metrics.

`runner.py` sets `complete=True` when the answer stage returns, including when
it reaches its answer cap; it does not test correctness or structural validity.
Thoughts ending at the thought cap or with an EOS before the closing delimiter
receive an inserted closing delimiter followed by answer generation. B's eleven
completed requests comprise two natural thought endings and nine forced endings
(six at the thought cap and three at early EOS). All eleven recorded answer
stages ended with `stop`, not `length`. Errors and deadline interruptions do not
count as complete. The historical `natural` counter requires both natural
thought closure and answer `stop`. `cap_reached` can overlap unfinished requests.

Throughput is actual generated thought plus answer tokens divided by generation
wall seconds across four GPUs; manually inserted delimiters are excluded.
`progress` windows are binned by token-arrival monotonic time, including any
last short bin. Request token counts saved on cancellation may omit tokens not
yet yielded by the engine. Use the recorded aggregate token meter for throughput.

## Files and provenance

- `runner.py`, `measurement.py`, `supervisor.py`, `core.py`, `runtime.py`,
  `telemetry.py`, `tiny_engine.py`, `cpu_fixture.py`: original S3 implementation.
- `system_prompt.txt`, `chat_template.jinja`: prompt and CPU template; GPU uses
  the model tokenizer's own template as in the historical runner.
- `kernel/kernel.py`: **empty-payload template**, byte-identical to
  `supervisor.py`. The original kernel embedded competition data and is excluded.
  Do not run this template directly: first prepare a private kernel below.
- `kernel/kernel-metadata.json`: historical GPU attachment/image metadata,
  not an upload instruction. Offline runtime and model attachment names remain
  references; their availability and server-side byte identity are not certified.
- `prepare_original.py`: unchanged historical S3 packaging code, retained for
  provenance; its development-relative paths are not the portable workflow.
- `prepare.py`: portable selection and private-kernel builder replacing the
  development builder's fixed repository paths. Does not run on import.
- `aggregate.py`: whitelist-only export from private raw records.
- `results/summary.json`: token counts, completion, closure, timing and speed.
  Anonymous request metrics have no seed, identifier or cross-arm linkage and
  are sorted by retained metrics, not original task name or admission order.
  Unadmitted requests contribute to unfinished counts but have no invented rows.
- `source_provenance.json`: hashes of unchanged development source files and
  explicit publication modifications; no data/prompt fingerprints.
- `references/`: model revision/files and offline runtime lock/manifest, reused
  from the final bundles (same pinned runtime). No wheels or weights are included.

## Reconstructing the selected workload privately

The original universe was the frozen development diagnostic output list used
with the saved B_exp009 NVARC candidate store (78 diagnostic outputs were used
in the replay, not a random draw of the entire public evaluation set). Exact
sampling requires the same frozen list and the same ranked top-two baseline
grids; regenerating baseline predictions can change the selected set. These
inputs and the selected IDs are deliberately not distributed here. Consequently
this release reproduces the protocol and reported aggregate arithmetic, but
does not independently recover the exact historical selected set from public
competition tasks alone.

Selection, implemented in `prepare.select`, is inherited from the earlier fast
measurement: exclude outputs without baseline candidates; compare the public
evaluation label with the saved top two; sort misses and hits separately by
output key; use `random.Random(20261022)` to sample twelve misses, then four hits,
then shuffle the concatenated list with the same RNG. S3 reused this frozen
selection unchanged. The two CPU-control outputs are the shortest full prompts
from different tasks, breaking ties by key. The GPU runner repeats the sixteen
rows for seed 0, then seed 1. Evaluation labels are used only in selection;
labels and baseline predictions are never included in the measurement payload.
This is development selection using public answers, not held-out evaluation.

Supply your own local input files outside this publication bundle:

- `outputs.json`: `[{"task": ..., "idx": ..., "key": ...}, ...]`, or an object
  with an `outputs` list; preserve the historical universe to reproduce selection.
- `baseline_top2.json`: map each output key to its ranked list of up to two grids.
  In the original development tree this was exported from saved inference
  outputs using `dev/planB/compare.py:load_baseline`; this release does not
  recompute rankings from raw model scores.
- an evaluation directory containing one task JSON per task, including public
  test labels. Obtain data separately under its applicable terms.

From this directory (Python 3.11, standard library only for preparation):

```bash
python3 -B prepare.py --outputs ../private-input/outputs.json \
  --baseline ../private-input/baseline_top2.json \
  --evaluation ../private-input/evaluation --output ../private-s3
```

The generated kernels in `../private-s3/kernel/` and `kernel_cpu/` **embed task
inputs and IDs** and must remain private. The builder refuses output under this
publication directory. These are packaging artifacts, not a GPU test. For an
actual rerun, use the historical offline runtime/model attachments or separately
provision the pinned versions and weights, and run the generated GPU kernel on
four L4s in an appropriate offline environment. CPU control uses the tiny Qwen3
fixture; it cannot validate 27B throughput, FP8, MTP or CUDA behavior. No automatic
GPU run, package installation, network request or upload is provided here.

## Recreating the published aggregates

Given a private raw S3 output directory containing `result.json` and B/E/F records:

```bash
python3 -B aggregate.py --input ../private-s3-output --output ../anonymous-summary.json
```

The exporter reads only allowlisted metrics and validates categorical values.
It excludes prompt hashes, token tails, paths, stack traces and error strings
as well as all task identifiers and text. Counts can be checked directly from
`results/summary.json`: B = 2 + 6 + 3 = 11; E = 0; aggregate tokens / generation
seconds rounds to 236 and 418. B has 27 admitted and 5 unadmitted requests;
E has all 32 admitted. We do not infer correctness from these records.

## Licensing

Original measurement and publication utility code: MIT-0, see `LICENSE`.
The ARC Prize Foundation prompt retains its MIT notice in
`licenses/MIT-ARC-Prize-Foundation.txt`. The Qwen template retains upstream
Apache-2.0 terms (`licenses/Apache-2.0.txt`). Model and runtime distributions
retain their respective licenses; no such distributions are bundled.
