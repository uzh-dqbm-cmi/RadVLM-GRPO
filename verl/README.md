# verl for RadVLM-GRPO

We than the contributors of verl for their awesome work!

## changes to original code
We modified / added files, including:
- this README.md
- added custom rewards in custom_rewards/
- added some scripts in scripts/
- added "naive_pool.py" and the relevant changes to add this into the verl code.
- added jobs in jobs/
- added examples/data_preprocess/mimic_mm.py

All these instructions assume that you are in this folder / relative to this position:
```
cd RadVLM-GRPO/verl/
```

## custom rewards location
Added custom rewards here, those can be used by verl

```
custom_rewards/
custom_rewards/radcliq_reward.py
custom_rewards/bertscore_reward.py
custom_rewards/google_bleu_reward.py
custom_rewards/radgraph_reward.py
```

Radcliq, bertscore, and radgraph rewards rely on models which can be started up like described in the following files, note that the job scripts themselves already start them correctly.

```
custom_rewards/reward_server/
custom_rewards/reward_server/launch_workers.sh
custom_rewards/reward_server/worker_radcliq.py
# and similar the ones for radgraph and bertscore
```

## reward manager location
Added this reward manager such that we can call the reward models in parallel.
```
verl/workers/reward_manager/naive_pool.py
```
This is set by
```
reward_model.reward_manager=naive_pool \
```

## Data parquet creation

Using your llava dataset created with RadVLM repo, you can create the parquet data files by executing the following command (change as you like):
```
python examples/data_preprocess/mimic_mm.py /path/to/dataset.json --local_dir SCRATCH/<username>/data/mimic
```

to enabled reasoning add --reasoning, which sets the "data_source" in the partquet to indicate reasoning.
```
python examples/data_preprocess/mimic_mm.py /path/to/dataset.json --local_dir SCRATCH/<username>/data/mimic --reasoning
```

to enabled reasoning add --reasoning, which sets the "data_source" in the partquet to indicate reasoning. But not add the /think, for example when training qwen3 vl directly without any SFT, i.e. no /think suffix,
```
python examples/data_preprocess/mimic_mm.py /path/to/dataset.json --local_dir SCRATCH/<username>/data/mimic --reasoning --no_think_suffix
```

## Slurm script 

The job scripts are in 

```
jobs/
```

For example 
```
jobs/mimic_radvlm_instruct_radcliq.sh
```

will get 9 nodes (8 for RL, 1 for radcliq reward) and call

```
jobs/mimic_radvlm_instruct_radcliq.sh
```

and update these:

```
WORK_DIR=/users/user/repos/RadVLM-GRPO/verl
```

the scripts might also rely on other paths like the path to the GREEN repo or the RadEval repo so you need to change them too so you can install them.


And similarily in the respective *_run.sh update these:
```
export WORK_DIR=/users/user/repos/RadVLM-GRPO/verl
export DATA_DIR=$SCRATCH/
```

This script will setup the ray master on node 0, the 32 radcliq reward model on node 1, and the remaining 7 nodes are ray worker nodes. Other scrips like the google bleu will only request 8 nodes. Since this one sets up radcliq it will also install RadEval. Note that you might have to change paths and so on in the scripts based on where you put your files and folders.

Note that these scripts are written to run on clariden (CSCS Alps) and if you are running them somewhere else you might have to make changes, potentially large ones.

## Checkpoint conversion to HF
After training with verl you can convert to HF with this script

```
bash scripts/merge.sh /path/to/global_step_300 informative_suffix
```

This is for convenience:
```
bash scripts/merge_multiple.sh
```



