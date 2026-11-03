# ARC-AGI-2 2026 — exp096 (publication draft)

This source bundle was prepared offline.
The selected exp096 submission is 57646082. Planned publication is after 2026-11-02 23:59 UTC
and by 2026-11-09 23:59 UTC; no public upload is performed by this bundle.

## Method and layout

NVARC's public 4B grid model performs task-local LoRA test-time training (TTT),
DFS generation and candidate ranking. GPUs are released before a separate
Qwen3.8-27B FP8 vLLM process. Complete valid Qwen answers may fill/replace a
second attempt when NVARC candidates exist. When NVARC has no candidate, Qwen
fills attempt_1. Failure or timeout preserves the available NVARC candidates.
TTT is the only training we run in this submitted solution; we use the public
NVARC pretrained/SFT weights as supplied. We do not claim to reproduce their
original pretraining, whose source/training-data completeness is unconfirmed.

- `notebook/kernel.py`: byte-for-byte local submitted script snapshot.
- `notebook/kernel-metadata.json`: unchanged private/offline attachment snapshot.
- `src/`: all 28 decoded payload members, including third-party material.
- `entry.py`, `rebuild.py`: first-party entry template and offline exact rebuild.
- `provenance.json`: hashes, file origins, permissions and original paths.
- `references/`: model identity and wheel-lock metadata, no weights/wheels/data.
- `LICENSE`, `NOTICE`, `evidence/`: first-party dual license and third-party audit.

Original kernel SHA-256: `0c00938e7de3e068bc11ebbe72a7a7e6c1081b090f6e19f453182d461ea05730`.
Payload SHA-256: `a25239f02cb7c0a93fcbe98276a5a30072075001dcabd84d3a8c2c11f903b7bd`.

## Rebuild without running inference

From this directory, with Python 3.11 and the standard library:

```bash
python3 -B rebuild.py --verify-only
# To also materialize the two verified files:
python3 -B rebuild.py
```

Without `--verify-only`, this writes `rebuilt/kernel.py` and `rebuilt/kernel-metadata.json` inside this
bundle, and verifies every member, payload and exact kernel hash. It does not
import any solver, create a GPU engine, install packages or access a network.
Do not run the original development build.py: it writes outside this bundle.
Different Python/zlib versions may fail byte-identical ZIP verification;
the unchanged `notebook/kernel.py` is the authoritative snapshot.

## Required external inputs

Use the official competition environment only after accepting its Rules:
https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2

Attach both model instances and both notebook outputs exactly as listed:

1. NVARC 4B: https://www.kaggle.com/models/sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1 — version 1.
2. Qwen 27B FP8: https://www.kaggle.com/models/mikedan7/qwen3-8-27b-fp8-official/pytorch/hf-fp8/1 — version 1; official revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`.
3. Unsloth/Flash patch output: https://www.kaggle.com/code/sorokin/pip-install-unsloth-flash-patch.
4. CPython 3.11 wheel runtime output: https://www.kaggle.com/code/sophie555/arc26-qwen38-runtime-cp311.
5. Competition source `arc-prize-2026-arc-agi-2`; no separate dataset sources.

The two notebook-output version numbers are not pinned in the submitted
metadata and remain unverified (未確認). Exact dependency-output reproduction
has therefore not been established. The local 180-wheel reference manifest SHA-256 is
`c1c04d8b9c1f2cab7b0fb420a085c021697191c72b2a49f0be12d5ea473296dd`; the runtime code locates a manifest by `vllm==0.19.1` and installs
its hashed requirements offline. Do not silently substitute a later output.
No dependency installation or download was performed during this preparation.

## Compute and execution

The submitted metadata pins the Docker image digest
`320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868`
and requests `machine_shape=NvidiaL4` (Kaggle L4 x4). Internet must remain OFF.
The saved runtime reference reports Python 3.11.13, PyTorch 2.10.0+cu128,
CUDA 12.8, vLLM 0.19.1 and Transformers 5.8.0. NVARC uses the separately
attached Unsloth environment; its full final component manifest is not saved.
Allow enough working storage for a 6.81 GB compressed wheel set, an unpacked
venv, model mounts and transient candidates; a measured peak disk/RAM value
is not established in this review. Four L4 GPUs provide 24 GB VRAM each;
Qwen uses tensor parallel size 4, FP8 weights/KV and MTP3 initially.

Once external inputs are supplied and a GPU run is separately authorized,
execute `notebook/kernel.py` in that Kaggle environment. It produces
`/kaggle/working/submission.json`. This document describes reproduction;
it does not authorize a push, GPU run or competition submission.

The script selects `arc-agi_test_challenges.json` when Kaggle sets
`KAGGLE_IS_COMPETITION_RERUN`; otherwise it uses evaluation challenges in a
short commit/debug mode. A local debug run is not a full hidden-test rerun.
Keep the submitted defaults for reproduction. `COMBO_OUT` and `COMBO_TEMP`
override output/temp directories. `COMBO_COMP_DIR` affects the entry/supervisor
input lookup only; NVARC starter.py and arc_solver.py independently resolve the
two fixed Kaggle competition mount paths. It does not fully redirect NVARC to
an arbitrary competition directory. This bundle embeds no personal local paths.
Do not publish generated challenge copies, submissions, per-task logs or caches.

## Time allocation and decoding

All deadlines are measured from entry, including setup:

| Event | exp096 default |
|---|---|
| NVARC soft stop | 3h / 10,800 s |
| NVARC cleanup / recovery hard end | 3h05m / 11,100 s |
| Qwen phase window | 8h30m after worst-case cleanup; setup/generation share this window |
| Qwen stop | 11h35m / 41,700 s |
| Final save | 11h40m / 42,000 s |
| Platform limit | 12h; 20m final margin |

Qwen: xhigh, thought cap 49,152 tokens, answer cap 8,192, concurrency 16,
FP8 KV and MTP3 initially. Forced continuation asks for the final grid after a
thought-cap/EOS interruption. Invalid, partial, truncated and cancelled outputs
are rejected. Retry behavior is preserved in runner.py/supervisor.py; retries
consume the same phase window. exp095 selects up to 128 outputs; exp096 selects
all outputs, with missing/zero NVARC candidates first. exp096 permits environment
overrides for NVARC duration/TTT and Qwen concurrency; the exact default is
3h / TTT on / 16. TTT uses 16 color permutations x 8 spatial transformations = 128 augmented
training items per task (with training-example order shuffled), 8 generation
spatial views and 4 rescoring views in these sources.

Commit/debug mode has short stops at 40m / 42m / 57m / 58m30s, uses NVARC's
four fixed debug task IDs and Qwen's two-output limit / 2,048 thought-token cap.
Those IDs are source routing logic; no task grid or solution is embedded.
This preparation checked syntax, hashes and packaging only. It did not rerun
TTT, model inference, GPU throughput, scoring or full 12-hour execution.

## Licensing and data handling

Our contributions: **MIT-0 OR CC-BY-4.0**, at the user's choice. Third-party code,
model-derived tokenizer files and template retain their original conditions.
Read NOTICE and provenance.json. The NVARC/yiheng and sorokin public notebook
code is treated as Apache-2.0 using Kaggle staff's general public-notebook policy;
their individual License fields were not verified. The ARC Prize Foundation
system prompt is MIT, with its copyright and full permission notice in
licenses/MIT-ARC-Prize-Foundation.txt. Complete Apache-2.0 terms are attached;
NOTICE identifies our three modified NVARC files and retains attribution.
Modification notices were added to `src/arc_decoder.py`, `src/arc_solver.py`, and
`src/starter.py`. The embedded payload in `notebook/kernel.py` remains byte-for-byte
identical to the scored submission snapshot. The scored server-side bytes have not
been independently retrieved and compared. `rebuild.py` removes only these exact
leading notices before checking original member hashes and rebuilding the payload.
`src/manifest.json` and provenance member hashes describe the original payload bytes.

The mikedan7 Kaggle Model hf-fp8 version 1 is Apache-2.0. Distribution licenses
alone do not resolve OSAID training-system compliance. Model weights and wheels
are referenced, not redistributed; package licenses remain separate. The selected-submission linkage is recorded below; attached Notebook-output
versions remain unverified (未確認).
Competition JSON, answers, ID selection fixtures and prediction records are
excluded from this bundle. Runtime-generated outputs are a separate concern:
the locally saved commit outputs contain problem-derived text and prediction
grids (internal audit: ../output_audit2.md). Before making a Notebook public,
confirm removal/exclusion of such Output artifacts, not only visible cell output.
Do not publish generated challenge copies, records, submissions or caches.

## Final selection linkage

| Selected ref | Local configuration | Notebook version (local record) | Local kernel SHA-256 | Attached dependency-output versions |
|---|---|---|---|---|
| 57646082 | Qwen-led / exp096 | `sophie555/arc2-exp096-qwen-main` v1 | `0c00938e7de3e068bc11ebbe72a7a7e6c1081b090f6e19f453182d461ea05730` | Both unverified (未確認) |

The final two selections, 57787093 (exp095) and 57646082 (exp096), were
confirmed via the SDK on 2026-11-02. The local record supplies the Notebook version.
The SHA-256 above was computed from the local development kernel.py and matches
this bundle's notebook/kernel.py byte for byte. The scored server-side kernel
bytes have not been independently retrieved and compared. The attached outputs
of sorokin/pip-install-unsloth-flash-patch and
sophie555/arc26-qwen38-runtime-cp311 both have unverified versions (未確認):
the saved attachment metadata names their sources without version pins.
