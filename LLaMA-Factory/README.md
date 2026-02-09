To perform instruction-finetuning, navigate to the yaml file at the path 
```
examples/train_full/qwen3vl_full_sft.yaml
```
You will be able to set up the finetuning parameters. 

The `dataset` argument is referenced in the `LLaMA-Factory/data/dataset_info.json` file, where the paths to RadVLM and cold-start instruction datasets have to be provided. 

The `tokenized_path` argument points to the tokenized dataset obtained after running the finetuning script (below) without using GPUs. It is indeed recommended to first tokenize the dataset, and then perform finetuning using the tokenized path.

You can run the finetuning script by executing the following command:

``` sbatch jobs/finetune.sh ```
