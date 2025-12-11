# Copyright 2025  –  Apache-2.0
import json, argparse, os, re, datasets
from tqdm import tqdm
from pathlib import Path
from verl.utils.hdfs_io import copy, makedirs
import pandas as pd

ABILITY = "radiology"
PROMPT_SUFFIXE = " /think"

MIN_PIXELS = 1024
MAX_PIXELS = 451584

def make_map_fn(split, reasoning, no_think_suffix):
    if reasoning:
        data_source = "mimic_grpo_reasoning"
    else:
        data_source = "mimic_grpo"

    def proc(example, idx):
        img_entry = {
            "image": f"file://{example['image']}",
            "min_pixels": MIN_PIXELS,
            "max_pixels":  MAX_PIXELS,
        }
        
        content = example["conversations"][0]["value"] 

        if reasoning and (not no_think_suffix):
            content = content + PROMPT_SUFFIXE

        prompt_msg = {
            "role": "user",
            "content": content
        }

        return {
            "data_source" : data_source,
            "prompt"      : [prompt_msg],
            "images": [img_entry],     
            "ability"     : ABILITY,
            "reward_model": {
                "style": "rule",
                "ground_truth": example["conversations"][1]["value"]
            },
            "extra_info"  : {
                "id":  example["id"],
                "split": split,
                "index": idx
            },
        }
    return proc

def main(json_path, local_dir, hdfs_dir=None, train_ratio=0.99, reasoning=False, no_think_suffix=False):
    with open(json_path) as f:
        rows = json.load(f)

    ds = datasets.Dataset.from_list(rows).shuffle(seed=42)
    n  = int(len(ds) * train_ratio)

    train_ds = ds.select(range(n)).map(make_map_fn("train", reasoning, no_think_suffix), with_indices=True, num_proc=8)
    val_ds   = ds.select(range(n, n+128)).map(make_map_fn("val", reasoning, no_think_suffix),   with_indices=True, num_proc=8)
    
    print(f"{len(train_ds)=}")
    print(f"{len(val_ds)=}")

    os.makedirs(local_dir, exist_ok=True)
    train_ds.to_parquet(os.path.join(local_dir, "train.parquet"))
    val_ds.to_parquet  (os.path.join(local_dir, "val.parquet"))

    if hdfs_dir:
        makedirs(hdfs_dir); copy(src=local_dir, dst=hdfs_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path")x
    parser.add_argument("--local_dir", default="~/data/chexpert_mm")
    parser.add_argument("--hdfs_dir",  default=None)
    parser.add_argument(
    '--reasoning',
    action='store_true',
    help='Enables reasoning and adds /think suffix unless disabled.'
    )
    parser.add_argument(
    '--no_think_suffix',
    action='store_true',
    help='Set this to true if its without the SFT, this will remove the /think suffix..'
    )

    main(**vars(parser.parse_args()))