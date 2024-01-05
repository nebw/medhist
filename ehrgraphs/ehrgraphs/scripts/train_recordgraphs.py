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
        EarlyStopping(
            monitor="valid/loss",
            min_delta=0.00000001,
            patience=args.training.patience,
            verbose=False,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    if args.training.write_predictions:
        callbacks.append(WritePredictionsDataFrame())

    if args.training.write_embeddings:
        callbacks.append(WriteRecordNodeEmbeddingsDataFrame())

    if args.training.write_attributions:
        callbacks.append(
            WriteFeatureAttributions(
                batch_size=args.datamodule.batch_size,
                baseline_mode=args.training.attribution_baseline_mode,
            )
        )

    trainer = Trainer(
        default_root_dir=output_root,
        logger=wandb_logger,
        callbacks=callbacks,
        **args.trainer,
    )

    trainer.fit(model, datamodule=datamodule)

    if hasattr(trainer, "checkpoint_callback"):
        OmegaConf.save(
            config=args, f=f"{Path(trainer.checkpoint_callback.dirpath).parent}/config.yaml"
        )
        wandb_logger.experiment.config[
            "best_checkpoint"
        ] = trainer.checkpoint_callback.best_model_path
        wandb_logger.experiment.config["best_score"] = trainer.checkpoint_callback.best_model_score


if __name__ == "__main__":
    main()
