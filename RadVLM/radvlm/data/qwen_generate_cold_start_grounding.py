#!/usr/bin/env python3
import argparse
import json
import pathlib
import os
import tempfile
from torch.utils.data import DataLoader, Subset
from concurrent.futures import ProcessPoolExecutor
import math

from radvlm.evaluation.models_loading_inference import inference_qwen2vl_vllm
from radvlm.data.datasets import Coldstart_Prompts
from radvlm.data.utils import custom_collate_fn


def parse_args():
    p = argparse.ArgumentParser(
        description="Run cold-start inference in parallel on a shuffled dataset and save each output as individual JSON files, skipping existing ones."
    )
    p.add_argument(
        "--json", required=True, type=str,
        help="Input dataset in JSON format"
    )
    p.add_argument(
        "--model_name", required=True, type=str,
        help="Model name or path"
    )
    p.add_argument(
        "--num_samples", required=True, type=int,
        help="Number of samples to process"
    )
    p.add_argument(
        "--out", required=True, type=str, 
        help="Output folder to write JSON into")
    p.add_argument(
        "--num_chunks", type=int, default=1, help="Total number of chunks to split the dataset into"
        )
    p.add_argument(
        "--chunk_id", type=int, default=0, help="0-based id of the chunk to run"
        )
    
    return p.parse_args()


THINK_SUFFIX = " /think"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def to_conversation(rec: dict) -> dict:
    """Use the 'instr' string as the human prompt verbatim; ensure it starts with '<image>\\n'."""
    img = rec.get("img_path")
    human = rec.get("txt", "")
    return {
        "image": img,
        "conversations": [
            {"from": "human", "value": human + THINK_SUFFIX},
            {"from": "gpt", "value": rec.get("full_output", "")},
        ],
        "id": rec.get("id"),
    }

def get_chunk_indices(n: int, num_chunks: int, chunk_id: int):
    if num_chunks <= 0:
        raise ValueError("num_chunks must be >= 1")
    if not (0 <= chunk_id < num_chunks):
        raise ValueError(f"chunk_id must be in [0, {num_chunks-1}], got {chunk_id}")
    chunk_size = math.ceil(n / num_chunks)
    start = chunk_id * chunk_size
    end = min(start + chunk_size, n)
    return list(range(start, end))


def main():
    args = parse_args()

    TEMPLATE_PATH = "/capstor/scratch/cscs/ndeperr/code/RadVLM-r1/RadVLM/radvlm/data/prompt_coldstart.txt"
    template_text = pathlib.Path(TEMPLATE_PATH).read_text(encoding="utf")

    dataset = Coldstart_Prompts(args.json, template_text)

    indices = get_chunk_indices(len(dataset), args.num_chunks, args.chunk_id)
    subset = Subset(dataset, indices)
    data_loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )

    print(f"Selected {len(indices)} examples: chunk {args.chunk_id+1}/{args.num_chunks} "
      f"({indices[0]}..{indices[-1]} inclusive)" if indices else
      f"Selected 0 examples: chunk {args.chunk_id+1}/{args.num_chunks}")

    with tempfile.TemporaryDirectory() as tmpdir:
        test_dataset_path = os.path.join(tmpdir, "test.parquet")

        output = inference_qwen2vl_vllm(
            data_loader=data_loader,
            model_name=args.model_name,
            process_batch_num=args.num_samples,
            r1=False,
            temperature=0.7,
            max_input_tokens=4096,
            max_new_tokens=8192,
            gpu_memory_utilization=0.87,
            test_dataset_path=test_dataset_path,
        )

    print(output)

    # Parallel conversion using all available CPUs
    max_workers = os.cpu_count() or 1
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        converted = list(ex.map(to_conversation, output))


    from pathlib import Path
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # simple filename:
    out_path = out_dir / f"cold_start_{args.chunk_id}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=4)

    print(f"Wrote {len(converted)} records to {out_path}")


if __name__ == "__main__":
    main()