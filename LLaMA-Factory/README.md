To perform instruction-finetuning, navigate to the yaml file at the path 
```
examples/train_full/qwen3vl_full_sft.yaml
```
You will be able to set up the finetuning parameters. 

The `dataset` argument is referenced in the `LLaMA-Factory/data/dataset_info.json` file, where the paths to RadVLM and cold-start instruction datasets have to be provided. 

You can run the finetuning script by executing the following command:

``` sbatch jobs/finetune.sh ```
