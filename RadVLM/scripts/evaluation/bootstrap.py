from pathlib import Path
import argparse
import json
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str,)
    args = parser.parse_args()
    return args

def load_jsonl(file):
    records = []
    with open(file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records

def load_json(file):
    with open(file, "r") as f:
        return json.load(f)

def load_output_file(dir_path: str):
    p = Path(dir_path)
    radcliq_file = next(p.glob("*_output_radcliq.json"), None)
    if radcliq_file is None:
        raise FileNotFoundError("No *_output_radcliq.json file found in directory.")

    racliq_records = load_jsonl(radcliq_file)


    green_file = next(p.glob("*_output_green.json"), None)

    green_dict = load_json(green_file)

    return racliq_records, green_dict

def bootstrap_mean(scores, n_boot=2000, rng_seed=0):
    scores = np.asarray(scores)
    n = scores.shape[0]
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = scores[idx]
    means = samples.mean(axis=1)
    return means

def do_bootstrap(scores, name):
    scores = np.array(scores, dtype=float)
    boot_means = bootstrap_mean(scores, n_boot=2000)

    point_estimate = scores.mean()
    lower = np.percentile(boot_means, 2.5)
    upper = np.percentile(boot_means, 97.5)

    print(f"Mean {name} score:", point_estimate)
    print("Bootstrap 95% CI:", (lower, upper))

if __name__ == "__main__":
    args = get_args()
    radcliq_records, green_dict = load_output_file(dir_path=args.dir)

    mean_radcliq = radcliq_records[0]

    radcliq_records = radcliq_records[1:]

    radcliq_scores = []

    for i, radcliq_record in enumerate(radcliq_records):
        assert radcliq_record["idx"] == i
        radcliq_scores.append(radcliq_record["radcliq"])

    do_bootstrap(radcliq_scores, "RadCliQ")
    

    green_scores = green_dict["green_scores"]
    do_bootstrap(green_scores, "GREEN")




