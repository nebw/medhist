#!/bin/bash

#SBATCH --job-name=identity_covariates      # Specify job name
#SBATCH --partition=gpu        # Specify partition name
#SBATCH --nodes=1              # Specify number of nodes
#SBATCH --mem=64GB                # Use entire memory of node
#SBATCH --gres=gpu:1           # Generic resources; 1 GPU
#SBATCH --time=48:00:00        # Set a limit on the total run time

script_path='/home/steinfej/code/ehrgraphs/ehrgraphs/scripts/train_recordgraphs.py'
config_path='/home/steinfej/code/ehrgraphs/config/'

partition=$SLURM_ARRAY_TASK_ID
tag=220306
model=covariates
covariates=agesex
name=$tag$model$covariates$partition

mamba activate /home/steinfej/miniconda3/envs/ehrgraphs

$script_path --config-path $config_path setup.entity=cardiors setup.project=RecordGraphs setup.name=$name datamodule.partition=$partition 'datamodule/covariates='$covariates model=$model head=mlp head.dropout=0.7 head.kwargs="{num_hidden:4096, num_layers:2, detach_clf:False}" training.binarize_records=True training.gradient_checkpointing=False training.write_predictions=True training.write_embeddings=False training.write_attributions=False
