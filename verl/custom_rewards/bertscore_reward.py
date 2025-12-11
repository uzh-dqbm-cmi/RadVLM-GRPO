import os
import re
import requests
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import random

MARK = "</think>"

REWARD_NODE_IP = os.environ["REWARD_NODE_IP"]
WORKER_BASE_PORT = int(os.environ["WORKER_BASE_PORT"])
WORKER_NUM_GPUS = int(os.environ["WORKER_NUM_GPUS"])
WORKER_INSTANCES_PER_GPU = int(os.environ["WORKER_INSTANCES_PER_GPU"])

NUM_WORKERS = WORKER_NUM_GPUS * WORKER_INSTANCES_PER_GPU

ports = [WORKER_BASE_PORT + i for i in range(NUM_WORKERS)]

print(f"{ports=}")

endpoints = [f"http://{REWARD_NODE_IP}:{port}" for port in ports]
n = len(endpoints)

_counter_lock = threading.Lock()

class _Counter:
    def __init__(self):
        self.value = 0

_counter = _Counter()

def _next_endpoint():
    with _counter_lock:
        i = _counter.value
        _counter.value = i + 1
    return endpoints[i % n]

def bertscore_score(pred_text, true_text, retries=3, timeout=5):
    backoff = 0.1

    for attempt in range(retries + 1):
        url = f"{_next_endpoint()}/score_bertscore"
        try:
            r = requests.post(
                url,
                json={"pred_text": pred_text, "true_text": true_text},
                timeout=timeout
            )
            if r.status_code == 200:
                return r.json()["score"]
            if r.status_code in (429, 500, 502, 503, 504):
                if attempt < retries:
                    time.sleep(backoff + random.random() * backoff)
                    backoff *= 2
                    continue
            r.raise_for_status()
        except (requests.ConnectionError, requests.Timeout):
            if attempt < retries:
                time.sleep(backoff + random.random() * backoff)
                backoff *= 2
                continue
            raise
    raise RuntimeError("All retries failed")

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if data_source == "mimic_grpo":
        return bertscore_score(solution_str, ground_truth)
    elif data_source == "mimic_grpo_reasoning":
        i = solution_str.rfind(MARK)

        if i == -1:
            return -0.0
        else:
            core_answer = solution_str[i + len(MARK):].strip()

            return bertscore_score(core_answer, ground_truth)

    else:
        assert False, "incorrect data source."

if __name__ == "__main__":
    ref = "et tube terminates 2 cm above the carina retraction by several centimeters is recommended for more optimal placement bibasilar consolidations better assessed on concurrent chest ct"
    hyp = "endotracheal tube terminates 2 5 cm above the carina bibasilar opacities likely represent atelectasis or aspiration"
    score = compute_score(data_source="mimic_grpo", solution_str=hyp, ground_truth=ref)
    print(score)