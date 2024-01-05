#!/usr/bin/env python3

import sys
from pathlib import Path
from socket import gethostname

import hydra
from hydra._internal.utils import get_args_parser
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch_geometric import seed_everything

from ehrgraphs.models.supervised import RecordsIdentityTraining
from ehrgraphs.training import setup_training
from ehrgraphs.utils.callbacks import (
    WriteFeatureAttributions,
    WritePredictionsDataFrame,
    WriteRecordNodeEmbeddingsDataFrame,
)
from ehrgraphs.utils.helpers import extract_metadata


@hydra.main("../../config", config_name="config")
def main(args: DictConfig):
    seed_everything(0)

    print(OmegaConf.to_yaml(args))

    host = gethostname()
    cluster = "charite-hpc" if host.startswith("s-sc") else "eils-hpc"
    output_root = f"{args.setup.root[cluster]}/{args.setup.output_path}"

    datamodule, model, tags = setup_training(args)

    wandb_logger = WandbLogger(
        entity=args.setup.entity,
        project=args.setup.project,
        group=args.setup.group,
        name=args.setup.name,
        tags=tags,
        config=args,
    )
    wandb_logger.watch(model, log_graph=True)

    records, covariates, endpoints = extract_metadata(datamodule)
    wandb_logger.experiment.config.update(
        {
            "feature_metadata": {
                "features": {"n": len(records) + len(covariates), "names": records + covariates},
                "records": {"n": len(records), "names": records},
                "covariates": {"n": len(covariates), "names": covariates},
            },
            "endpoint_metadata": {"n": len(endpoints), "names": endpoints},
        }
    )

    callbacks = [
        ModelCheckpoint(mode="min", monitor="valid/loss", save_top_k=1, save_last=True),
    ]

    trainer = Trainer(
        default_root_dir=output_root,
        logger=wandb_logger,
        callbacks=callbacks,
        **args.trainer,
    )
    trainer.model = model
    trainer.datamodule = datamodule
    trainer.checkpoint_callback.on_pretrain_routine_start(trainer, model)
    trainer.checkpoint_callback.best_model_path = args.inference.checkpoint_path

    predictions_callback = WritePredictionsDataFrame()
    predictions_callback.on_fit_end(trainer, model)

    embeddings_callback = WriteRecordNodeEmbeddingsDataFrame()
    embeddings_callback.on_fit_end(trainer, model)


if __name__ == "__main__":
    main()
