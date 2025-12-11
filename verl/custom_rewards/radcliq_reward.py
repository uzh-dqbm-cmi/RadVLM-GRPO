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

def radcliq_score(pred_text, true_text, retries=3, timeout=5):
    backoff = 0.1

    for attempt in range(retries + 1):
        url = f"{_next_endpoint()}/score_radcliq"
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
        return radcliq_score(solution_str, ground_truth)
    elif data_source == "mimic_grpo_reasoning":
        i = solution_str.rfind(MARK)

        if i == -1:
            return -3
        else:
            core_answer = solution_str[i + len(MARK):].strip()

            return radcliq_score(core_answer, ground_truth)

    else:
        assert False, "incorrect data source."



# 64-thread throughput test for 30 seconds 
_success_lock = threading.Lock()
_attempts = 0
_successes = 0
_failures = 0

def _worker(stop_time, sample_timeout=2):
    global _attempts, _successes, _failures
    while time.time() < stop_time:
        try:
            with _success_lock:
                _attempts += 1

            _ = score("predicted text", "true text", retries=1, timeout=sample_timeout)
            with _success_lock:
                _successes += 1
        except Exception:
            with _success_lock:
                _failures += 1

def throughput_test(duration_s=30, concurrency=64):
    stop_time = time.time() + duration_s
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for _ in range(concurrency):
            ex.submit(_worker, stop_time)
    elapsed = duration_s

    with _success_lock:
        attempts = _attempts
        successes = _successes
        failures = _failures
    rps = successes / elapsed if elapsed > 0 else 0.0
    return {
        "duration_s": duration_s,
        "concurrency": concurrency,
        "num_workers": NUM_WORKERS,
        "attempts": attempts,
        "successes": successes,
        "failures": failures,
        "success_rps": rps,
    }

FACTOR = 500

if __name__ == "__main__":
    import RadEval
    radcliq = RadEval.CompositeMetric()

    refs = ["no acute cardiopulmonary abnormality" * FACTOR,
        "et tube terminates 2 cm above the carina retraction by several centimeters is recommended for more optimal placement bibasilar consolidations better assessed on concurrent chest ct"  * FACTOR,
        "there is no significant change since the previous exam the feeding tube and nasogastric tube have been removed"  * FACTOR,
        "unchanged mild pulmonary edema no radiographic evidence pneumonia"  * FACTOR,
        "no evidence of acute pulmonary process moderately large size hiatal hernia"  * FACTOR,
        "no acute intrathoracic process" * FACTOR]
    
    hyps = ["no acute cardiopulmonary abnormality"  * FACTOR,
            "endotracheal tube terminates 2 5 cm above the carina bibasilar opacities likely represent atelectasis or aspiration"  * FACTOR,
            "there is no significant change since the previous exam"  * FACTOR,
            "unchanged mild pulmonary edema and moderate cardiomegaly"  * FACTOR,
            "no evidence of acute cardiopulmonary process moderate hiatal hernia"  * FACTOR,
            "no acute cardiopulmonary process"  * FACTOR]

    precomputed = []
    for pred, true in zip(hyps, refs):
        _, rewards = radcliq.predict(hyps=[pred[:2048]], refs=[true[:2048]])
        precomputed.append(-float(rewards[0]))

    print(f"{precomputed=}")

    lock = threading.Lock()
    attempts = 0
    successes = 0
    failures = 0

    def _worker(stop_time):
        global attempts, successes, failures
        i = 0
        L = len(hyps)
        while time.time() < stop_time:
            pred = hyps[i]
            true = refs[i]
            expected = precomputed[i]
            api = float(radcliq_score(pred, true))
            with lock:
                attempts += 1
                if api == expected:
                    successes += 1
                else:
                    failures += 1
            i = (i + 1) % L

    duration_s = 30
    concurrency = 32
    stop_time = time.time() + duration_s
    threads = []
    for _ in range(concurrency):
        t = threading.Thread(target=_worker, args=(stop_time,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    rps = successes / duration_s if duration_s > 0 else 0.0
    print("== Reward server load test ==\n"
        f"Duration: {duration_s}s | Concurrency: {concurrency} | "
        f"Workers: {NUM_WORKERS}\n"
        f"Attempts: {attempts} | Successes: {successes} | "
        f"Failures: {failures}\n"
        f"Throughput (success RPS): {rps:.2f}")


    exit()
    stats = throughput_test(duration_s=30, concurrency=64)
    print(
        "== Reward server load test ==\n"
        f"Duration: {stats['duration_s']}s | Concurrency: {stats['concurrency']} | "
        f"Workers: {stats['num_workers']}\n"
        f"Attempts: {stats['attempts']} | Successes: {stats['successes']} | "
        f"Failures: {stats['failures']}\n"
        f"Throughput (success RPS): {stats['success_rps']:.2f}"
    )