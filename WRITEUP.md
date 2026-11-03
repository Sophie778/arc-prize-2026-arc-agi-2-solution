# Sharing a fixed inference budget between NVARC and Qwen

## Summary

Our ARC Prize 2026 ARC-AGI-2 solution combines NVARC's task-specific adaptation with direct grid prediction by Qwen3.8-27B. The final selected submissions are 57787093 (exp095, NVARC-led, public score 36.11) and 57646082 (exp096, Qwen-led, public score 34.03). They use the same model families but allocate time differently. These are public leaderboard scores; we do not report private results here.

ARC tasks require inferring transformations from a few demonstrations. [NVARC](https://www.kaggle.com/c/arc-prize-2025/writeups/nvarc) provides task-adapted grid prediction; [Qwen](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/tree/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a) supplies language-based reasoning. Combining solvers leaves a practical question: how should adaptation, reasoning, and answer collection share a fixed budget? We used public weights. Our contribution is integration and measurement of complementary answers and deadline-sensitive completion, rather than a new foundation model or a proven optimal allocation.

## Pipeline

NVARC supplies a public 4B grid model. For each task, low-rank adaptation (LoRA) test-time training (TTT) adapts the model using augmented demonstration pairs. Depth-first search generates candidate grids, which are then ranked using transformed views of the task. Our final configurations use eight generation views and four rescoring views. TTT is the only training performed by the submitted pipeline. Earlier experiments with additional synthetic-data training did not supply the weights used in these submissions.

Qwen3.8-27B predicts grids directly from demonstrations and test inputs represented as JSON. It runs through vLLM with FP8 weights, FP8 key/value (KV) cache, tensor parallelism across four GPUs, and multi-token prediction (MTP) speculative decoding. The final defaults use xhigh reasoning, a 49,152-token thought cap, an 8,192-token answer cap, and concurrency 16. When reasoning hits its cap or stops before producing an answer, a continuation asks the model to finish with the output grid. Incomplete, invalid, truncated, and cancelled answers are rejected.

The integration retains NVARC's leading candidate when available and lets a complete Qwen answer fill or replace the second attempt. In the Qwen-led configuration, Qwen supplies the first attempt when NVARC has no candidate. Missing NVARC candidates receive priority in Qwen's queue. Failures and timeouts preserve the available NVARC predictions. Thus, the system can use complementary answers without requiring a learned judge to choose which model is correct.

Both configurations run sequentially on L4×4. All deadlines below are relative to program entry. The Qwen window starts after worst-case cleanup and includes setup and generation; it is not uninterrupted generation time. Selecting an output for the queue does not guarantee completion.

| Setting | NVARC-led: exp095 | Qwen-led: exp096 |
|---|---|---|
| NVARC stop | 8h10m | 3h |
| Cleanup/recovery hard end | 8h15m | 3h05m |
| Qwen window after cleanup | 3h20m | 8h30m |
| Qwen target limit | 128 outputs | All outputs |
| Qwen stop | 11h35m | 11h35m |
| Final save | 11h40m | 11h40m |
| Platform limit | 12h | 12h |

NVARC releases the GPUs before Qwen starts in a separate process, separating conflicting dependency environments. The supervisor bounds execution, recovers available candidates, and targets final saving 20 minutes before the platform limit.

```text
Demonstrations + test inputs
  -> NVARC: LoRA TTT -> candidate generation -> rescoring
  -> saved candidates -> GPU cleanup -> Qwen queue
                         (missing candidates first)
  -> Qwen: reasoning -> answer continuation -> validation
  -> merge with saved NVARC candidates -> two attempts/output
Deadline: stop Qwen 11h35m -> save 11h40m -> limit 12h
```

## Conditions and observations

D16 is deliberately selected: 12 NVARC errors and four successes, not a random sample. The sustained-speed workload repeats D16 twice: 32 requests for 16 distinct outputs. D32 is a separate random sample of 32 public-evaluation outputs after excluding D16, reused during development. Correctness means exact-match outputs; throughput is aggregate tokens/second. Counts from different sets are not pooled. The sustained-speed comparison uses low reasoning with a 24,576-token thought cap; the D32 direct-answer runs (JSON/packed and direct repeats) use xhigh with a 49,152-token thought cap. In the table, Colab G4 denotes one RTX PRO 6000 Blackwell GPU with 96 GB VRAM and tensor parallelism 1 (TP1); L4×4 runs use TP4.

| Comparison | Set | Hardware | KV | Concurrency | Window | Runs/condition | Correct |
|---|---|---|---|---|---|---|---|
| low / xhigh | D16 | L4×4 | BF16 / FP8 | 16 / 16 | ~51 / ~61 min elapsed | 1 | 2/16 / 5/16 |
| MTP off / on | D16 | L4×4 | BF16 | 16 | First 600s generation | 1 | Not measured |
| Sustained speed: BF16 KV, concurrency 16 / FP8 KV, concurrency 32 | D16 twice | L4×4 | BF16 / FP8 | 16 / 32 | 30 min generation | 1 | Not reported |
| JSON / packed | D32 | L4×4 | FP8 | 32 / 16 | 90 min generation each | 1 | 12/32 / 3/32 |
| Python long | D32 | L4×4 | FP8 | 32 | 90 min generation | 1 | 6/32 |
| Python without timeout | D32 | Colab G4 | BF16 | 32 | No timeout | 1 | 9/32 |
| Python short/repair | D32 | Colab G4 | BF16 | 32 | Stopped ~54 min elapsed | 1 | 3/32 |
| Direct repeats | D32 | Colab G4 | BF16 | 32 | Full completion | 3 | 6/32, 11/32, 8/32 |

Earlier saved-output replay on a separate 78-output diagnostic set combined stored NVARC candidates with Qwen results: 22/78 became 24/78, gaining two and losing none. This supports complementarity in that earlier integration condition, not a measured leaderboard gain for the final pipeline.

Both D16 conditions prompted for an answer after interrupted reasoning. Low reasoning with a 24,576-token cap solved 2/16; xhigh with 49,152 solved 5/16. In the latter, three of 12 answers requested after the cap were correct. Effort, cap, KV precision, and runtime settings changed together. This motivated the recipe but does not isolate continuation prompting.

MTP increased throughput from 207 to 311 tokens/second, or 1.50×, during the first 600 seconds. We retained it; this short-window measurement is not an end-to-end acceleration factor or an isolated leaderboard contribution.

The sustained-speed comparison exposes a different tradeoff. In the same 30-minute window, BF16 KV/concurrency 16 completed 11/32 requests at 236 tokens/second; FP8 KV/concurrency 32 completed none at 418. Higher aggregate throughput did not mean earlier completed answers. KV precision and concurrency changed together.

| Measured endpoint at 30 minutes | BF16 KV, concurrency 16 | FP8 KV, concurrency 32 |
|---|---|---|
| Aggregate tokens/second | 236 | 418 |
| Completed requests | 11/32 | 0/32 |

These are measured endpoints, not a reconstructed completion curve or an extrapolation.

## What did not work

Packing grids into dimensions and digit rows reduced input tokens by 62%, but correct D32 outputs fell from 12/32 to 3/32. We retained JSON. Concurrency also changed; harder spatial interpretation is a hypothesis, not an established mechanism.

Python generation with demonstration checking did not exceed direct answering's correct-output count on D32 under the tested conditions. One L4×4 run (FP8 KV, concurrency 32, 90-minute generation window) solved 6/32 versus direct prediction's 12/32. A Colab G4 variant (BF16 KV, concurrency 32, no timeout) solved 9/32. A short-reasoning repair variant there solved 3/32, with seven unfinished after approximately 54 minutes elapsed. Each ran once. Hardware, parallelism, and stopping conditions differ; this does not establish general inferiority of program synthesis.

Single Qwen-led submissions with thought caps of 65,536 and 32,768 scored 27.36 and 28.61. We retained 49,152 without claiming optimality: extra reasoning competes with output coverage, and run variation obscures small differences.

## Run-to-run variance

Two scored runs with unchanged contents gave 34.31 and 36.11 for NVARC-led, and 34.03 and 30.97 for Qwen-led. These observed ranges (about 1.8 and 3.06 points) are neither standard deviations nor significance thresholds. Selecting different allocations retains methodological diversity; private-test complementarity was not measured.

Three complete Colab G4 direct-answer runs on D32 under the same configuration (BF16 KV, concurrency 32) solved 6, 11, and 8 outputs. Their oracle union was 15/32; vote-selected top-two answers achieved 13/32. Saved-result simulation retried only outputs whose first reasoning reached its cap. It estimated 3.1 additional correct outputs per million retry tokens, versus 5.6 correct outputs per million first-attempt tokens. Retried outputs are a harder selected group. This simulation informed our choice to prioritize first-attempt coverage; it does not establish a general advantage over retries.

## Time budget accounting

The recorded test workload has 240 tasks and 259 requested outputs. TTT is task-specific; completion is output-specific. Planning estimates for NVARC-led suggested processing about 170 tasks, leaving 76–92 outputs without NVARC candidates, while Qwen might answer about 60. Simple subtraction leaves 16–32 outputs. These are estimates, not audited completion counts from the selected scored run.

The number of unanswered outputs in the Qwen-led configuration remains unverified. The NVARC-led estimate suggests coverage may constrain queue-order improvements: Qwen might spend its window on missing-candidate outputs. That hypothesis informed our decision against an order change. Testing it requires completion logs from full scored runs.

## Evaluation and configuration selection

We used the public evaluation set for configuration selection, and these results are not independent held-out estimates. Its distribution informed earlier synthetic-data generation and training experiments. We repeatedly reused the 32-output development subset to choose grid formatting, alternative models, Python-based answering, and retry policies. There was also a design for selecting TTT amounts using likelihoods of known evaluation answers. The submitted pipeline uses public NVARC weights with TTT on the demonstrations supplied for each task; it does not consult hidden test-output answers.

A separate queue-order experiment compared confidence measures using correctness on a 57-task diagnostic subset drawn from public evaluation. Public leaderboard scores also informed version and final-submission selection. These uses introduce selection bias even when individual experiments start from randomly sampled development outputs. We therefore keep local exact-match counts, speed measurements, and public leaderboard scores distinct. None supplies an independent estimate of the final system's performance on unseen tasks, and local improvements should not be converted directly into leaderboard points.

## Reproduction

Only the GitHub code package, without outputs, is planned for publication at `<GITHUB_URL>`. The submitted Notebooks and their outputs will not be published. The package contains local snapshots of the two configurations, source scripts, rebuild tools, provenance, and dependency references. It excludes competition grids, answers, prediction records, reasoning logs, and weights. READMEs describe official competition inputs, model mounts, and offline dependencies. Running these snapshots reproduces a local configuration, not an already verified copy of a selected scored artifact.

The table links the selected submissions to Notebook versions recorded in the local experiment ledger and to SHA-256 hashes computed from the locally saved kernel.py files. Each local kernel matches its bundled snapshot byte for byte. The scored server-side kernel bytes have not been independently retrieved and compared. The versions of the two attached dependency Notebook outputs (sorokin/pip-install-unsloth-flash-patch and sophie555/arc26-qwen38-runtime-cp311) remain unverified: the saved metadata names their sources without version pins. Rebuild verification checks source/payload identity without inference; it does not reproduce a score or upstream NVARC training. Reduced debug runs do not validate full-budget coverage.

| Selected ref | Local configuration | Notebook version (local record) | Local kernel SHA-256 | Attached dependency-output versions |
|---|---|---|---|---|
| 57787093 | NVARC-led / exp095 | `sophie555/arc2-exp095-qwen-force` v1 | `64fe8ff6397a4c796baf3ed498a348334cb822f95eb8391ba64bd9e8f951116b` | Both unverified (未確認) |
| 57646082 | Qwen-led / exp096 | `sophie555/arc2-exp096-qwen-main` v1 | `0c00938e7de3e068bc11ebbe72a7a7e6c1081b090f6e19f453182d461ea05730` | Both unverified (未確認) |

## Licenses and attribution

We thank Ivan Sorokin and Jean-François Puget for [NVARC](https://www.kaggle.com/c/arc-prize-2025/writeups/nvarc), yiheng for the [reproduction notebook](https://www.kaggle.com/code/yiheng/reproduce-nvarc-2025-results), and sorokin for the [Unsloth/Flash dependency notebook](https://www.kaggle.com/code/sorokin/pip-install-unsloth-flash-patch) and public weights. Our package treats the public notebook code as Apache-2.0 under Kaggle staff's general public-notebook policy; the individual notebook License fields were not verified. This does not establish a repository-wide license for upstream NVARC. Modification notices and inherited attribution are retained.

We also credit Qwen for the model and chat template, mikedan7 for the FP8 Kaggle mirror, the ARC Prize Foundation for its MIT-licensed system prompt, and phamtruongdat for MIT-0 runtime preparation procedures. Our original contributions are offered under MIT-0 OR CC-BY-4.0; third-party components retain their own terms. The referenced Qwen distribution is Apache-2.0. Distribution permissions do not establish compliance of upstream training systems with every competition requirement. The release records preserve that unresolved distinction and the component-specific notices.

## Answer to the budget question

Within this budget, preserving NVARC candidates and adding validated Qwen answers offered complementary successes in saved-output replay. Our measurements favor checking completed answers alongside token throughput when allocating time. They support the integration strategy, while leaving the best allocation and final-run coverage unresolved.
