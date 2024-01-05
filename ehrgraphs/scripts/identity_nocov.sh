#!/bin/bash

#SBATCH --job-name=identity_covariates      # Specify job name
#SBATCH --partition=gpu        # Specify partition name
#SBATCH --nodes=1              # Specify number of nodes
#SBATCH --mem=128GB                # Use entire memory of node
#SBATCH --gres=gpu:1           # Generic resources; 1 GPU
#SBATCH --time=48:00:00        # Set a limit on the total run time

script_path='/home/wildb/dev/projects/ehrgraphs/ehrgraphs/scripts/train_recordgraphs.py'
experiment='best_identity_220428_datafix_220624'

partition=$SLURM_ARRAY_TASK_ID
tag=220627
model=identity
covariates=no_covariates
name=$tag$model$covariates$partition_nofinalbias

conda activate /home/wildb/envs/gnn/

$script_path datamodule.partition=$partition training.write_predictions=True setup.name=$name experiment=$experiment
