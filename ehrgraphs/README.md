# RecordGraphs

## Setup pre-commit

Automatic code formatting, removal of unused imports, and much more on commits:

```
pip install pre-commit
pre-commit install --install-hooks
```

## Training examples

Training runs can be started and configured via the training script `ehrgraphs/scripts/train_recordgraphs.py`. Examples:

1. Train on first two GPUs with default options:

    > `./train_recordgraphs.py 'trainer.gpus=[0, 1]'`

2. Test run with small dataset and most common endpoints:

    > `./train_recordgraphs.py setup=test 'datamodule.use_top_n_phecodes=10'`

3. Configure model type:

    > `./train_recordgraphs.py model=gnn`

    > `./train_recordgraphs.py model=identity`

    > `./train_recordgraphs.py model=graph_embeddings`

    > `./train_recordgraphs.py model=learned_embeddings`

    > `./train_recordgraphs.py model=covariates`

4. Configure head type:

    > `./train_recordgraphs.py head=linear`

    > `./train_recordgraphs.py head=mlp`

5. Train with covariates:

    > `./train_recordgraphs.py datamodule.covariates=agesex`
