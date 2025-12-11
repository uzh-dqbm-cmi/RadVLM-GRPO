#!/usr/bin/env python3
import argparse, json, os, random, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests
import numpy as np

# adjust if needed
WORKER_BASE_PORT = 9400
WORKER_NUM_GPUS = 4
WORKER_INSTANCES_PER_GPU = 8
NUM_WORKERS = WORKER_NUM_GPUS * WORKER_INSTANCES_PER_GPU

# you need to set this! (hostname -i)
REWARD_NODE_IP = "172.28.43.100"

MAX_THREADS = 32

ports = [WORKER_BASE_PORT + i for i in range(NUM_WORKERS)]
endpoints = [f"http://{REWARD_NODE_IP}:{port}" for port in ports]
n = len(endpoints)

_counter_lock = threading.Lock()
_counter = 0

def _next_endpoint():
    global _counter
    with _counter_lock:
        i = _counter
        _counter = i + 1
    return endpoints[i % n]

def radcliq_score(pred_text, true_text, timeout=5):
    url = f"{_next_endpoint()}/score_radcliq"
    if pred_text == "" or true_text == "":
        return -3.0

    r = requests.post(
        url,
        json={"pred_text": pred_text, "true_text": true_text},
        timeout=timeout
    )
    if r.status_code == 200:
        return r.json()["score"]

    raise RuntimeError("something went wrong with radcliq score")

def main():
    parser = argparse.ArgumentParser(description="Compute RadCliQ scores for a JSON list.")
    parser.add_argument("input_json", type=Path, help="Path to JSON file (list of dicts).")
    parser.add_argument("--threads", type=int, default=MAX_THREADS, help="Thread pool size.")
    args = parser.parse_args()

    with args.input_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit("Input must be a JSON array of objects.")

    # Build tasks: (index, pred_text, true_text)
    tasks = [(i, item.get("output", ""), item.get("answer", "")) for i, item in enumerate(data)]

    print(f"Scoring {len(tasks)} items across {len(endpoints)} endpoints with {args.threads} threads...")

    scores = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        future_to_idx = {
            ex.submit(radcliq_score, pred, true): idx
            for idx, pred, true in tasks
        }
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]

            score = fut.result()
            scores[idx] = score
            if idx and idx % 1000 == 0:
                print(f"...scored ~{idx} items")

    # Attach scores back to data
    for i, s in enumerate(scores):
        data[i]["radcliq"] = s

    # Report means
    arr = np.array([s for s in scores if s is not None], dtype=float)
    mean_incl_fail = float(arr.mean()) if arr.size else float("nan")

    print(f"RadCliQ mean: {mean_incl_fail:.4f}")

    results_path = args.input_json
    output_dir = os.path.dirname(results_path)
    base_name = os.path.basename(results_path)

    if base_name.endswith("_output.json"):
        output_filename = base_name.replace("_output.json", "_output_radcliq.json")
    else:
        output_filename = os.path.splitext(base_name)[0] + "_radcliq.json"

    output_path = os.path.join(output_dir, output_filename)

 

    # Optional output
    with open(output_path, "w", encoding="utf-8") as out:
        out.write(json.dumps({"mean_radcliq": mean_incl_fail}, ensure_ascii=False) + "\n")
        for item in data:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote scored records to {output_path}")

if __name__ == "__main__":
    main()
