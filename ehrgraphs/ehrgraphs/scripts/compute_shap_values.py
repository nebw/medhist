#!/usr/bin/env python3

import argparse
import datetime

import captum.attr
import hydra
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from hydra import compose, initialize
from more_itertools import nth
from omegaconf import OmegaConf
from tqdm import tqdm

import ehrgraphs
from ehrgraphs.models.supervised import Batch, RecordsIdentityTraining
from ehrgraphs.training import setup_training


class ShapWrapper(nn.Module):
    def __init__(self, lightning_module, n_records, device):
        super().__init__()

        self.lightning_module = lightning_module
        self.lightning_module.module_dict["head"].gradient_checkpointing = False

        self.n_records = n_records
        self.device = device

    def forward(self, batch_data):
        head_outputs = self.lightning_module.module_dict["head"](batch_data)

        if self.lightning_module.use_endpoint_embeddings:
            head_outputs["logits"] = torch.einsum(
                "lg,bg->bl", label_node_embeddings, head_outputs["logits"]
            )

        return head_outputs["logits"]


# %%
def compute_shap(
    config_path: str,
    checkpoint_path: str,
    cmd_overrides: list,
    device: str,
    batch_idx: int,
    endpoint: str,
    perturbations_per_eval: int,
    n_samples: int,
    tag: str,
    output_path: str,
    model_id: str,
):
    n_samples *= perturbations_per_eval

    with hydra.initialize(config_path=config_path):
        cfg = hydra.compose(config_name="config", overrides=cmd_overrides)

    datamodule, model, _ = setup_training(cfg)

    modules = []
    modules.append(["head", model.module_dict["head"]])

    ckpt_model = RecordsIdentityTraining.load_from_checkpoint(
        checkpoint_path, modules=torch.nn.ModuleDict(modules)
    )
    ckpt_model.eval()
    ckpt_model.freeze()

    test_loader = datamodule.test_dataloader()
    batch = nth(test_loader, batch_idx)

    num_records = len(datamodule.record_cols)
    net = ShapWrapper(ckpt_model.to(device), num_records, device)

    shapley = captum.attr.ShapleyValueSampling(net)

    target_idx = datamodule.labels.index(endpoint)

    attrs = []
    for i in tqdm(range(len(batch.records))):
        input = batch.records[[i]].bool().float().to(device)
        baseline_data = torch.zeros(1, input.shape[1]).to(device)

        nonzero_idxs = np.argwhere(input[0].cpu().numpy() != 0)[:, 0]
        zero_idxs = np.argwhere(input[0].cpu().numpy() == 0)[:, 0]

        features_masks = torch.zeros_like(input).long()
        features_masks[0, nonzero_idxs] = torch.from_numpy(np.arange(len(nonzero_idxs)) + 1).to(
            device
        )

        attr = (
            shapley.attribute(
                input,
                baselines=baseline_data,
                feature_mask=features_masks,
                n_samples=n_samples,
                perturbations_per_eval=perturbations_per_eval,
                target=target_idx,
            )
            .detach()
            .cpu()[0]
        )
        attr[zero_idxs] = torch.nan

        attrs.append(attr)

    attr = torch.stack(attrs)

    attr_df = pd.DataFrame(
        attr.cpu().numpy(), columns=datamodule.record_cols_input, index=batch.eids.cpu().numpy()
    )
    attr_df.index.name = "eid"

    attr_df.reset_index().to_feather(
        f"{output_path}/{tag}_attribution_shaplysampling_{model_id}_{n_samples}_{endpoint}_{batch_idx}.feather"
    )


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/sc-projects/sc-proj-ukb-cvd/results/models/RecordGraphs/34r1puue/checkpoints/epoch=15-step=12159.ckpt",
        required=False,
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="34r1puue",
        required=False,
    )
    parser.add_argument(
        "--config_path", type=str, default="dev/projects/ehrgraphs/config", required=False
    )
    parser.add_argument(
        "--cmd_overrides",
        type=list,
        default=("experiment=best_identity_220428_datafix_220624",),
        required=False,
    )
    parser.add_argument("--device", type=str, default="cuda:0", required=False)
    parser.add_argument("--batch_idx", type=int, default=0, required=False)
    parser.add_argument("--endpoint", type=str, default="phecode_008", required=False)
    parser.add_argument("--perturbations_per_eval", type=int, default=32, required=False)
    parser.add_argument("--n_samples", type=int, default=100, required=False)
    parser.add_argument(
        "--tag", type=str, default=datetime.date.today().strftime("%y%m%d"), required=False
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/sc-projects/sc-proj-ukb-cvd/results/recordgraphs/attributions",
        required=False,
    )

    args = parser.parse_args()

    compute_shap(**vars(args))
