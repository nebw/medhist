import copy
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torchmetrics
import wandb
from pytorch_lightning import LightningModule
from torch.optim import AdamW

from ehrgraphs.data.collate import Batch
from ehrgraphs.models.loss_wrapper import LossWrapper
from ehrgraphs.modules.attention import PoolingMultiheadAttention
from ehrgraphs.modules.head import AlphaHead
from ehrgraphs.schedulers.alpha_schedulers import AlphaScheduler


class RecordsTraining(LightningModule):
    def __init__(
        self,
        modules: torch.nn.ModuleDict,
        record_weights: Optional[torch.Tensor] = None,
        losses: Optional[Iterable[LossWrapper]] = [],
        label_mapping: Optional[Dict] = dict(),
        incidence_mapping: Optional[Dict] = dict(),
        optimizer: Optional[torch.optim.Optimizer] = AdamW,
        optimizer_kwargs: Optional[Dict] = {"weight_decay": 5e-4},
        metrics_list: Optional[List[torchmetrics.Metric]] = [torchmetrics.AUROC],
        metrics_kwargs: Optional[List[Dict]] = None,
        exclusions_on_metrics: Optional[bool] = True,
        normalize_node_embeddings: bool = True,
        alpha_scheduler: Optional[AlphaScheduler] = None,
        node_dropout: Optional[float] = None,
        binarize_records: bool = False,
        use_endpoint_embeddings: bool = False,
        test_time_augmentation_steps: int = 1,
        use_lr_scheduler: bool = False,
        record_weights_learnable: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters(
            "label_mapping",
            "incidence_mapping",
            "optimizer_kwargs",
            "exclusions_on_metrics",
            "normalize_node_embeddings",
            "node_dropout",
            "binarize_records",
            "use_endpoint_embeddings",
            "test_time_augmentation_steps",
            "use_lr_scheduler",
            "record_weights_learnable",
        )

        self.module_dict = modules

        self.record_weights = record_weights
        if record_weights_learnable:
            self.record_weights = torch.nn.Parameter(self.record_weights)

        self.losses = losses

        self.label_mapping = label_mapping
        self.incidence_mapping = incidence_mapping

        if isinstance(self.module_dict["head"], AlphaHead):
            alpha = alpha_scheduler.get_alpha()
            self.module_dict["head"].update_alpha(alpha)

        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs

        self.valid_metrics = self.initialize_metrics(label_mapping, metrics_list, metrics_kwargs)
        self.train_metrics = copy.deepcopy(self.valid_metrics)
        self.max_mean_metrics = defaultdict(float)
        self.exclusions_on_metrics = exclusions_on_metrics

        self.normalize_node_embeddings = normalize_node_embeddings

        self.executor = ThreadPoolExecutor(max_workers=32)
        self.alpha_scheduler = alpha_scheduler

        self.node_dropout = node_dropout
        self.binarize_records = binarize_records
        self.use_endpoint_embeddings = use_endpoint_embeddings

        self.test_time_augmentation_steps = test_time_augmentation_steps
        self.use_lr_scheduler = use_lr_scheduler

    def initialize_metrics(
        self,
        label_mapping: Dict,
        metrics_list: List[torchmetrics.Metric],
        metrics_kwargs: List[Dict],
    ):
        """
        Soft-wrapper for metrics instatiation. When loading from cpt these are already instatiated,
        throwing a typeerror when called -> soft wrap it!
        :return:
        """
        if metrics_kwargs is None:
            metrics_kwargs = [{} for m in metrics_list]

        metrics = torch.nn.ModuleDict()
        for l in label_mapping.keys():
            metrics[l] = torch.nn.ModuleList(
                [m(**kwargs) for m, kwargs in zip(metrics_list, metrics_kwargs)]
            )
        return metrics

    def update_metrics(self, metrics: Iterable, batch: Batch, outputs: Dict, loss_dict: Dict):
        """
        Calculate the validation metrics. Stepwise!
        :return:
        """
        times = batch.times.clone()
        no_event_idxs = times == 0
        times[no_event_idxs] = batch.censorings[:, None].repeat(1, times.shape[1])[no_event_idxs]

        for idx, (_, m_list) in enumerate(metrics.items()):
            p_i = outputs["head_outputs"]["logits"][:, idx].detach().cpu().float()
            l_i = batch.events[:, idx].cpu()
            t_i = times[:, idx].cpu()

            if self.exclusions_on_metrics:
                mask = batch.exclusions[:, idx] == 0
                p_i = p_i[mask]
                l_i = l_i[mask]
                t_i = t_i[mask]

            for m in m_list:
                if m.__class__.__name__ == "CIndex":
                    m.update(p_i, l_i.long(), t_i)
                else:
                    m.update(p_i, l_i.long())

    def compute_and_log_metrics(self, metrics: Dict, kind: Optional[str] = "train"):
        if kind != "train":
            averages = defaultdict(list)
            incidence_averages = defaultdict(list)

            def compute(args):
                l, m_list = args
                rs = []

                for m in m_list:
                    r = m.compute()
                    rs.append(r)

                return rs

            results = list(self.executor.map(compute, metrics.items()))

            for (l, m_list), rs in zip(metrics.items(), results):
                for m, r in zip(m_list, rs):
                    self.log(f"{kind}/{self.label_mapping[l]}_{m.__class__.__name__}", r)
                    averages[m.__class__.__name__].append(r)
                    incidence_averages[self.incidence_mapping[l]].append(r)
                    m.reset()

            for m, v in averages.items():
                v = torch.stack(v)
                # TODO: filter out nans, e.g. endpoints with no positive samples
                # probably should be done in a better way, otherwise CV results might not
                # be valid.
                metric_name = f"{kind}/mean_{m}"
                value = torch.nanmean(v)
                self.log(metric_name, value)

                self.logger.experiment.log(
                    {
                        f"{kind}/auroc_hist": wandb.Histogram(
                            sequence=v[torch.isfinite(v)].cpu(), num_bins=100
                        )
                    }
                )

                if value > self.max_mean_metrics[metric_name]:
                    self.max_mean_metrics[metric_name] = value

                self.log(f"{metric_name}_max", self.max_mean_metrics[metric_name])

            for m, v in incidence_averages.items():
                v = torch.stack(v)
                metric_name = f"{kind}/mean_{m}"
                value = torch.nanmean(v)
                self.log(metric_name, value)

        else:
            for l, m_list in metrics.items():
                for m in m_list:
                    m.reset()

    def shared_step(self, batch: Batch, n_average: int = 1) -> Dict:
        if n_average > 1:
            self.train()

        outputs = self(batch)
        if n_average > 1:
            head_logits = [outputs["head_outputs"]["logits"]]
            for _ in range(n_average - 1):
                head_logits.append(self(batch)["head_outputs"]["logits"])
            outputs["head_outputs"]["logits"] = torch.stack(head_logits).mean(dim=0)

        loss_dict = self.loss(batch, outputs)

        return outputs, loss_dict

    def training_step(self, batch: Batch, batch_idx: int):
        outputs, loss_dict = self.shared_step(batch)
        self.update_metrics(self.train_metrics, batch, outputs, loss_dict)
        if self.alpha_scheduler:
            self.alpha_scheduler.step()
        if isinstance(self.module_dict["head"], AlphaHead):
            alpha = self.alpha_scheduler.get_alpha()
            self.log("head_alpha", alpha, on_step=True)
            self.module_dict["head"].update_alpha(alpha)

        for k, v in loss_dict.items():
            self.log(
                f"train/{k}",
                v.item(),
                prog_bar=True,
                batch_size=batch.events.shape[0],
                sync_dist=True,
            )
        return loss_dict

    def training_epoch_end(self, outputs) -> None:
        self.compute_and_log_metrics(self.train_metrics, kind="train")

    def validation_step(self, batch: Tuple, batch_idx: int):
        outputs, loss_dict = self.shared_step(batch, n_average=self.test_time_augmentation_steps)
        self.update_metrics(self.valid_metrics, batch, outputs, loss_dict)

        for k, v in loss_dict.items():
            self.log(
                f"valid/{k}",
                v.item(),
                prog_bar=True,
                batch_size=batch.events.shape[0],
                sync_dist=True,
            )
        return {f"val_{k}": v for k, v in loss_dict.items()}

    def validation_epoch_end(self, outputs) -> None:
        self.compute_and_log_metrics(self.valid_metrics, kind="valid")

    def maybe_concat_covariates(self, batch: Batch, features: torch.Tensor):
        if features is None:
            assert batch.covariates is not None
            return batch.covariates

        if batch.covariates is not None:
            features = torch.cat((features, batch.covariates), axis=1)

        return features

    def get_projection(self, head_outputs: torch.Tensor):
        if "projector" in self.module_dict:
            return self.module_dict["projector"](head_outputs["pre_logits"])
        else:
            return None

    def forward(self, batch: Batch) -> Dict:
        record_node_embeddings, label_node_embeddings = self.get_record_node_embeddings(batch)

        individual_features, future_features = self.get_individual_features(
            batch, record_node_embeddings
        )
        individual_features = self.maybe_concat_covariates(batch, individual_features)

        head_outputs = self.module_dict["head"](individual_features)
        head_projection = self.get_projection(head_outputs)

        if "endpoint_embedding_head" in self.module_dict:
            label_node_embeddings = self.module_dict["endpoint_embedding_head"](
                label_node_embeddings
            )

        if self.use_endpoint_embeddings:
            head_outputs["logits"] = torch.einsum(
                "lg,bg->bl", label_node_embeddings, head_outputs["logits"]
            )

        return dict(
            record_node_embeddings=record_node_embeddings,
            individual_features=individual_features,
            future_features=future_features,
            head_outputs=head_outputs,
            head_projection=head_projection,
        )

    def loss(self, batch: Batch, outputs: Dict) -> Dict:
        loss_dict = {}

        losses = []
        for loss_wrapper in self.losses:
            loss, loss_unscaled = loss_wrapper.compute(batch, outputs)
            losses.append(loss)

            loss_dict[loss_wrapper.name] = loss_unscaled.detach()
            loss_dict[f"{loss_wrapper.name}_scaled"] = loss.detach()

        loss_dict["loss"] = torch.sum(torch.stack(losses))

        return loss_dict

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = self.optimizer(self.parameters(), **self.optimizer_kwargs)

        schedulers = []
        if self.use_lr_scheduler:
            reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.2,
                patience=2,
                min_lr=1e-6,
            )

            scheduler = {
                "scheduler": reduce_lr_on_plateau,
                "name": "ReduceLROnPlateau",
                "monitor": "train/loss",
            }

            schedulers.append(scheduler)

        return [optimizer], schedulers


class GraphEncoderMixin:
    def get_record_node_embeddings(self, batch: Batch):
        full_node_embeddings = self.module_dict["graph_encoder"](
            batch.graph.x_dict, batch.graph.adj_t_dict
        )
        record_node_embeddings = full_node_embeddings[batch.record_indices]

        if self.normalize_node_embeddings:
            record_node_embeddings = torch.nn.functional.normalize(record_node_embeddings)

        return record_node_embeddings, None


class ShuffledGraphEncoderMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # hardcoded for now, could be passed from datamodule.record_node_indices
        self.record_cols_idxer = np.arange(16201)
        np.random.shuffle(self.record_cols_idxer)

    def get_record_node_embeddings(self, batch: Batch):
        full_node_embeddings = self.module_dict["graph_encoder"](
            batch.graph.x_dict, batch.graph.adj_t_dict
        )
        record_node_embeddings = full_node_embeddings[batch.record_indices[self.record_cols_idxer]]

        if self.normalize_node_embeddings:
            record_node_embeddings = torch.nn.functional.normalize(record_node_embeddings)

        return record_node_embeddings, None


class LearnedEmbeddingsMixin:
    def get_record_node_embeddings(self, batch: Batch):
        record_indices = torch.arange(batch.records.shape[1], device=batch.records.device)
        record_node_embeddings = self.module_dict["embedder"](record_indices)

        if self.normalize_node_embeddings:
            record_node_embeddings = torch.nn.functional.normalize(record_node_embeddings)

        return record_node_embeddings, None


class PretrainedEmbeddingsMixin:
    def get_record_node_embeddings(self, batch: Batch):
        full_node_embeddings = batch.graph.x_dict["0"]
        record_node_embeddings = full_node_embeddings[batch.record_indices]
        label_node_embeddings = full_node_embeddings[batch.label_indices]

        if self.normalize_node_embeddings:
            record_node_embeddings = torch.nn.functional.normalize(record_node_embeddings)
            label_node_embeddings = torch.nn.functional.normalize(label_node_embeddings)

        return record_node_embeddings, label_node_embeddings


class RecordNodeFeaturesMixin:
    def get_individual_features(self, batch: Batch, record_node_embeddings: torch.Tensor):
        def get_features(records):
            with torch.autocast(dtype=torch.float32, device_type="cuda"):
                weighted_record_node_embeddings = (
                    torch.sigmoid(self.record_weights)[:, None].to(record_node_embeddings.device)
                    * record_node_embeddings
                )

                if self.binarize_records:
                    assert records.min() >= 0
                    records = torch.sign(records)

                if self.node_dropout is not None:
                    random_mask = (
                        (torch.rand(*records.shape) > self.node_dropout).to(records.device).float()
                    )
                    features = torch.einsum(
                        "bn,bn,nr->br", records, random_mask, weighted_record_node_embeddings
                    )
                else:
                    features = records @ weighted_record_node_embeddings

                if not self.binarize_records:
                    features = torch.sign(features) * torch.log1p(torch.abs(features))
            return features

        records_features = get_features(batch.records)

        future_features = None
        if "projector" in self.module_dict:
            future_features = get_features(batch.future_records)

        return records_features, future_features


class RecordsGraphTraining(GraphEncoderMixin, RecordNodeFeaturesMixin, RecordsTraining):
    pass


class RecordsShuffledGraphTraining(
    ShuffledGraphEncoderMixin, RecordNodeFeaturesMixin, RecordsTraining
):
    pass


class RecordsLearnedEmbeddingsTraining(
    LearnedEmbeddingsMixin, RecordNodeFeaturesMixin, RecordsTraining
):
    pass


class RecordsPretrainedEmbeddingsTraining(
    PretrainedEmbeddingsMixin, RecordNodeFeaturesMixin, RecordsTraining
):
    pass


class IdentityMixin:
    def get_record_node_embeddings(self, batch: Batch):
        full_node_embeddings = batch.graph.x_dict["0"]
        label_node_embeddings = full_node_embeddings[batch.label_indices]

        if self.normalize_node_embeddings:
            label_node_embeddings = torch.nn.functional.normalize(label_node_embeddings)

        return None, label_node_embeddings

    def get_individual_features(self, batch: Batch, record_node_embeddings: torch.Tensor):
        def get_features(records):
            if self.binarize_records:
                assert records.min() >= 0
                features = torch.sign(records)
            else:
                with torch.autocast(dtype=torch.float32, device_type="cuda"):
                    features = records
                    features = torch.sign(features) * torch.log1p(torch.abs(features))
            return features

        records_features = get_features(batch.records)

        future_features = None
        if "projector" in self.module_dict:
            future_features = get_features(batch.future_records)

        return records_features, future_features


class RecordsIdentityTraining(IdentityMixin, RecordsTraining):
    pass


class CovariatesOnlyMixin:
    def get_record_node_embeddings(self, batch: Batch):
        return None, None

    def get_individual_features(self, batch: Batch, record_node_embeddings: torch.Tensor):
        return (None, None)


class CovariatesOnlyTraining(CovariatesOnlyMixin, RecordsTraining):
    pass


class SelfAttentionEmbeddingsMixin:
    def __init__(self, embedding_dim_in, embedding_dim_out, num_heads, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pma = PoolingMultiheadAttention(embedding_dim_in, embedding_dim_out, num_heads)

    def get_individual_features(self, batch: Batch, record_node_embeddings: torch.Tensor):
        # only calculate attention for records with at least one sample in current batch
        features = torch.sign(batch.records)
        feature_mask = features.sum(dim=0) > 0
        attn_mask = (features > 0)[:, feature_mask]
        attn_features = (
            record_node_embeddings[feature_mask].unsqueeze(0).repeat(features.shape[0], 1, 1)
        )

        with torch.autocast(dtype=torch.float32, device_type="cuda"):
            individual_features = self.pma(attn_features, attn_mask)

        return individual_features, None


class RecordsSelfAttentionEmbeddingsTraining(
    SelfAttentionEmbeddingsMixin, PretrainedEmbeddingsMixin, RecordsTraining
):
    pass
