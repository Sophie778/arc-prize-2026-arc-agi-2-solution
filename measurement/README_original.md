> Historical measurement README; commands and paths refer to the development layout.
> The publication workflow is README.md; no historical data payload is included.
> The author-specific temporary environment path was replaced by python3.

# Qwen3.8-27B sustained speed measurement (S3, MIT-0)

`dev/qwen38_speed` のコピーを起点に変更。プロンプト・固定16件・chat template・モデル・環境分離・telemetryを継承。dev/qwen38_tp22 の2エンジン制御は不使用。正解ラベル・NVARC予測は読み込まない。

| 条件 | KV | MTP先読み数 | 同時 | 投入 |
|---|---|---:|---:|---:|
| B | auto (BF16) | 3 | 16 | 32 |
| E | fp8 | 3 | 32 | 32 |
| F | fp8 | 4 | 32 | 32 |

全条件：同じ16件×seed0/seed1、low、思考24,576、回答4,096、temperature1/top_p.95/top_k20。TP4、公式FP8重み、vLLM0.19.1、gpu_memory_utilization=.85、max_model_len65536、max_num_batched_tokens8192、prefix caching無効、CUDA Graph有効。E/Fで変えるのは先読み数だけ。条件ごとにプロセス・worker groupを停止してエンジンを再起動する。

全32件が自然終了・上限到達するまで、または生成1,800秒まで測る。従来版同様、思考上限・早期EOSからも回答段へ進め、強制継続の事実を記録する。エラーを自然終了に数えない。

FP8+MTP起動失敗（起動タイムアウトを含む）は元のrun.log・attemptsに保持し、MTPなしを起動する。通常の失敗は残った600秒枠、タイムアウト後は新たな最大600秒枠を与えるが、条件全体2,430秒と元の全体期限で制限する。そのため後退後の生成時間は短くなり得る。Eが失敗すればE/FともMTPなし。Fだけ失敗した場合、先行Eの記録をsuperseded_mtp_measurementに保持し、EをMTPなしで再測定する。この例外的な追加測定にも元の全体期限を適用するため、残時間不足なら未完となる。異なるMTP有無をE/F比較として扱わない。MTP重み検査に失敗した場合、Bは基準条件を変えずskipped、E/FはMTPなしとなる。

`/kaggle/working/qwen38_s3/result.json`、条件ごとの`run.log`・`engine_config.json`・`records`・`progress.json`を保存する。

- 持続速度：思考＋回答の実生成token総数÷生成開始から終了までの壁時間。手動で足した閉じタグは含めない。起動時間は別記。
- `progress.windows`：monotonic到着時刻の300秒区間ごとのtoken/s。最後の短い区間は実秒数で割り、complete_window=false。全件終了後の空時間は足さない。
- `metrics`：accepted/draftedと受理率、preemption累積数、KV使用率ピーク。ログ未取得はnull。KV容量はvLLMのGPU KV cache sizeログから抽出。
- `completion_counts`：natural、cap_reached、complete、unfinished（未投入も含む）。cap_reachedは未完と重複し得る。段別finish_reasonと早期EOSは各recordに保存する。
- `answer_comparison.json`：Bと同じkey・seedで両方自然終了した出力のみ、抽出盤面の完全一致・回答textの一致を記録。抽出失敗は一致に数えずnull。正解率・同等性の証明ではない。
- 強制kill時の速度は保存済みrecordと1秒ごとのtoken checkpointの大きい方による下限。partial_lower_bound=true。CPUではGPU/MTP計数はnull。

時間：親側の条件期限は条件開始から600+1,800+排出猶予30=2,430秒。runnerのcancel/abortが滞ってもcondition_deadlineでgroupをTERM→KILLし、回収して次条件へ進む。準備360 + 3×2,430 = 7,650秒。install最大3プロセス、各条件のprobeとrunner各1回の停止・排出を最大4秒ずつ保守的に加えた通常上限7,686秒（128分6秒）。後退や追加E再測定がある場合も以下の全体期限を優先する。全体8,070秒（134分30秒）で通常停止、独立したOSプロセスwatchdogが8,095秒（134分55秒）で全追跡groupと親をKILL。親異常終了時にもgroupをKILLする。2h15m=8,100秒を超えない。独立watchdog起動はpayload展開・installより前。異常停止時はhard_stop.json優先。Kaggle待機時間は含まない。

ディスク：重みをworkingにコピーしない。venvとキャッシュは`/kaggle/temp/sol_s3`。ログは子プロセスごと32MiBで打ち止め後も排出・計数を続ける。各出力段のtextは256Ki文字、checkpointは2,048token間隔。32×3=96件×安全側21回×最大3MiB ≈6.05GiB。例外のE再測定を足した128件でも約8.1GiB、ログ・payload・token meterを含め累積書込10GB未満を見積もる（上限20GB）。

再構築とGPUなしの確認：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 dev/qwen38_s3/prepare.py
PYTHONDONTWRITEBYTECODE=1 python3 dev/qwen38_s3/test_flow.py
PYTHONDONTWRITEBYTECODE=1 python3 dev/qwen38_s3/local_cpu_check.py
```

CPU kernelは3,392パラメータのQwen3をその場で作成し、本番と同じrunner/schedulerに通す。各周はB生成→Eのmodel load後の意図的起動失敗→MTPなしE生成→F生成→別runの生成中に期限停止、を3周。各正常条件2件・思考2/回答2token。これは実transformers forwardを使う制御試験で、27B・vLLM・FP8 KV・MTP・CUDAの確認ではない。独立OS watchdogは別のプロセステストで3周確認する。

`kernel/`はGPU用生成物（pushは今回の権限外）、`kernel_cpu/`のみprivate CPUでpushする。ハッシュはpayload_manifest.json。作業結果はwork/sol_s3_answer.md。
