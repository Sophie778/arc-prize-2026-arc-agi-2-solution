import os

# NOTE (fork, 2026-08-19): Kaggle has been flip-flopping the competition data
# mount between /kaggle/input/competitions/<slug>/ and /kaggle/input/<slug>/
# (both observed within 48h).  Resolve at runtime, never hardcode.
_COMP_DIRS = ["/kaggle/input/competitions/arc-prize-2026-arc-agi-2",
              "/kaggle/input/arc-prize-2026-arc-agi-2"]
COMP_DIR = next((d for d in _COMP_DIRS if os.path.isdir(d)), _COMP_DIRS[0])
import time
import json
import torch
import argparse
import torch.multiprocessing as mp


def local_worker(rank, queue, end_time):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    torch.set_default_device("cpu")

    # Fix Unsloth patching issue
    if rank > 0:
        while not os.path.exists(f"/kaggle/worker{rank-1}"):
            time.sleep(5)
    
    from arc_solver import worker

    with open(f"/kaggle/worker{rank}", "w") as f:
        f.write("Ok")
    
    print(f"[Rank {rank}] start!")
    
    worker(rank, queue, end_time)
    
    print(f"[Rank {rank}] done!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", type=float, default=0.0)
    args = parser.parse_args()

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

    if rerun_mode:
        test_path = COMP_DIR + "/arc-agi_test_challenges.json"
    else:
        test_path = COMP_DIR + "/arc-agi_evaluation_challenges.json"

    with open(test_path, "r") as f:
        data = json.load(f)

    queue = mp.Manager().Queue()

    for key in sorted(data.keys()):
        if not rerun_mode:
            if key not in ["0934a4d8", "36a08778", "981571dc", "aa4ec2a5"]:
                continue
        queue.put(key)
    for _ in range(4):
        queue.put(None)

    # NOTE (fork, 2026-08-20): upstream called mp.spawn(..., nprocs=4), which
    # joins unconditionally.  On exp009 V3 one worker hung inside model load and
    # never pulled a task; the other three finished all 60 tasks in 3.3h, but the
    # unconditional join held the kernel until the 12h limit killed it -- so the
    # submission cell never ran and no submission.json was produced.  In a scoring
    # rerun that is a zero, not a slow run.  Bound the wait by the same deadline
    # the workers already respect, and kill whatever is still alive, so the
    # notebook always reaches the submission step.
    ctx = mp.spawn(local_worker, args=(queue, args.end_time), nprocs=4, join=False)
    procs = ctx.processes
    last_report = 0.0
    while True:
        alive = [i for i, p in enumerate(procs) if p.is_alive()]
        if not alive:
            break
        now = time.time()
        if now > args.end_time:
            print(f"[starter] deadline reached, terminating stuck worker(s): {alive}",
                  flush=True)
            for i in alive:
                procs[i].terminate()
            break
        if now - last_report > 300:
            last_report = now
            try:
                pending = queue.qsize()
            except Exception:
                pending = -1
            # a silent worker was the failure mode: make liveness observable
            print(f"[starter] alive workers {alive}, queue {pending}", flush=True)
        time.sleep(10)
    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.kill()
            p.join(5)