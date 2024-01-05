from typing import Dict, Optional

import numpy as np
import torch

from ehrgraphs.data.collate import Batch
from ehrgraphs.loss.contrastive import nt_xent_loss


class LossWrapper:
    name: str
    scale: float

    def __init__(self, name: str, scale: float = 1.0):
        self.name = name
        self.scale = scale


class EndpointClassificationLoss(LossWrapper):
    def __init__(
        self,
        loss_fn,
        loss_fn_kwargs: dict = {},
        scale: float = 1.0,
        use_exclusion_mask: bool = True,
    ):
        super().__init__("BCE", scale)

        self.use_exclusion_mask = use_exclusion_mask
        self.loss_fn = loss_fn(**loss_fn_kwargs)

    def compute(self, batch: Batch, outputs: Dict) -> Dict:
        loss = self.loss_fn(outputs["head_outputs"]["logits"], batch.events)

        with torch.autocast(dtype=torch.float32, device_type="cuda"):
            if self.use_exclusion_mask:
                # TODO: hack
                exclusion_mask = batch.exclusion_mask.bool().float()
                inclusion_mask = 1 - exclusion_mask
                loss = (loss * inclusion_mask).sum(dim=0) / inclusion_mask.sum(dim=0)

            loss = loss[torch.isfinite(loss)].mean()

        return loss * self.scale, loss


class EndpointTTELoss(LossWrapper):
    def __init__(
        self,
        loss_fn,
        loss_fn_kwargs: dict = {},
        scale: float = 1.0,
        use_exclusion_mask: bool = True,
        loss_weights: Optional[np.array] = None,
    ):
        super().__init__("CoxPH", scale)

        self.use_exclusion_mask = use_exclusion_mask
        self.loss_fn = loss_fn(**loss_fn_kwargs)
        if loss_weights is not None:
            loss_weights = torch.from_numpy(loss_weights)[0]
        self.loss_weights = loss_weights

    def compute(self, batch: Batch, outputs: Dict) -> Dict:
        logits = outputs["head_outputs"]["logits"]

        # set time for 0 events to censoring time. TODO: might not be the best place to do this
        # here...
        # TODO: hack - epsilon s.t. censoring_time != death_time
        times = batch.times.clone()
        no_event_idxs = times == 0
        times[no_event_idxs] = batch.censorings[:, None].repeat(1, times.shape[1])[no_event_idxs]

        losses = []
        for i in range(logits.shape[1]):
            losses.append(self.loss_fn(logits[:, i], times[:, i], batch.events[:, i]))
        losses = torch.stack(losses)

        if self.loss_weights is not None:
            losses = losses * self.loss_weights[:, None].to(losses.device)

        loss = losses.mean()

        with torch.autocast(dtype=torch.float32, device_type="cuda"):
            if self.use_exclusion_mask:
                assert False
                # TODO: hack
                exclusion_mask = batch.exclusion_mask.bool().float()
                inclusion_mask = 1 - exclusion_mask
                loss = (loss * inclusion_mask).sum(dim=0) / inclusion_mask.sum(dim=0)

            loss = loss[torch.isfinite(loss)].mean()

        return loss * self.scale, loss


class EndpointContrastiveLoss(LossWrapper):
    def __init__(self, scale: float = 1.0):
        super().__init__("CPC", scale)

    def compute(self, batch: Batch, outputs: Dict) -> Dict:
        future_records = torch.nn.functional.normalize(outputs["future_features"], p=2, dim=-1)
        head_projection = torch.nn.functional.normalize(outputs["head_projection"], p=2, dim=-1)
        predictions_to = torch.einsum("ac,bc->ab", future_records, head_projection)
        contrastive_loss = nt_xent_loss(predictions_to).mean()

        return contrastive_loss * self.scale, contrastive_loss
