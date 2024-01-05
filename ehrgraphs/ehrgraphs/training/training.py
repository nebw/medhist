import datetime
import math
import warnings
from socket import gethostname

import hydra
import numpy as np
import pandas as pd
import torch
import torchmetrics
from omegaconf import DictConfig

from ehrgraphs.data.data import WandBGraphData, get_or_load_wandbdataobj
from ehrgraphs.data.datamodules import EHRGraphDataModule
from ehrgraphs.data.sampling import DummySampler, SubgraphNeighborSampler
from ehrgraphs.loss.focal import FocalBCEWithLogitsLoss
from ehrgraphs.loss.tte import CIndex, CoxPHLoss
from ehrgraphs.models.loss_wrapper import (
    EndpointClassificationLoss,
    EndpointContrastiveLoss,
    EndpointTTELoss,
)
from ehrgraphs.models.supervised import (
    CovariatesOnlyTraining,
    RecordsGraphTraining,
    RecordsIdentityTraining,
    RecordsLearnedEmbeddingsTraining,
    RecordsPretrainedEmbeddingsTraining,
    RecordsSelfAttentionEmbeddingsTraining,
    RecordsShuffledGraphTraining,
)
from ehrgraphs.models.vicreg import VICRegRecordsGraphTraining
from ehrgraphs.modules.gnn import HeteroGNN
from ehrgraphs.modules.head import AlphaHead, IndependentMLPHeads, LinearHead, MLPHead, ResMLPHead


def setup_training(args: DictConfig):
    tags = list(args.setup.tags)

    host = gethostname()
    cluster = "charite-hpc" if host.startswith("s-sc") else "eils-hpc"
    data_root = f"{args.setup.root[cluster]}/{args.setup.data_path}"

    if args.datamodule.t0_mode.startswith("fixed_date:"):
        _, date_str = args.datamodule.t0_mode.split(":")
        t0_mode = datetime.datetime.fromisoformat(date_str)
    else:
        t0_mode = args.datamodule.t0_mode
    del args.datamodule.t0_mode

    if args.setup.use_data_artifact_if_available:
        data = get_or_load_wandbdataobj(
            data_root,
            identifier=args.setup.data_identifier,
            entity=args.setup.entity,
            project=args.setup.project,
            **args.setup.data,
        )
    else:
        data = WandBGraphData(
            data_root=data_root,
            wandb_entity=args.setup.entity,
            wandb_project=args.setup.project,
            **args.setup.data,
        )

    # load node embeddings from file and replace embeddings in data
    if (
        args.datamodule.load_embeddings_path is not None
        and args.model.model_type != "LearnedEmbeddings"
        and args.model.model_type != "TransformerEmbeddings"
    ):
        embedding_df = pd.read_feather(args.datamodule.load_embeddings_path)
        node_id_lookup = {v: k for k, v in enumerate(data.graph.node_ids)}
        node_idxs = [node_id_lookup[n] for n in embedding_df.nodes]

        if len(node_idxs) != len(data.graph.x):
            print(
                f"Warning! Number of nodes ({len(data.graph.x)}) and corresponding embeddings"
                f" ({len(node_idxs)}) don't match."
            )

        data.graph.x[node_idxs] = torch.from_numpy(np.stack(embedding_df["embeddings"].values))[:]

    edge_weight_threshold = args.datamodule.edge_weight_threshold
    keep_edges = data.graph.edge_weight > edge_weight_threshold
    data.graph.edge_index = data.graph.edge_index[:, keep_edges]
    data.graph.edge_code = data.graph.edge_code[keep_edges]
    data.graph.edge_weight = data.graph.edge_weight[keep_edges]

    if args.datamodule.sampler.sampler_type == "DummySampler":
        graph_sampler = DummySampler

        graph_sampler_kwargs = {}
    elif args.datamodule.sampler.sampler_type == "SubgraphNeighborSampler":
        graph_sampler = SubgraphNeighborSampler

        graph_sampler_kwargs = dict(
            num_neighbors=[args.datamodule.sampler.neighbors_per_step]
            * args.datamodule.sampler.num_steps,
        )
    else:
        assert False

    datamodule = EHRGraphDataModule(
        data,
        graph_sampler=graph_sampler,
        graph_sampler_kwargs=graph_sampler_kwargs,
        t0_mode=t0_mode,
        **args.datamodule,
    )
    datamodule.prepare_data()

    def get_head(head_config, num_head_features):
        if args.datamodule.task == "binary":
            incidence = datamodule.train_dataset.labels_events.mean(
                axis=0
            )  # if this is not respecting potential exclusions, could it crash in the loss/metrics for rare events?
        else:
            incidence = None

        if head_config.model_type == "Linear":
            cls = LinearHead
        elif head_config.model_type == "MLP":
            cls = MLPHead
        elif head_config.model_type == "ResMLP":
            cls = ResMLPHead
        elif args.head.model_type == "MLP_independent":
            cls = IndependentMLPHeads
        elif head_config.model_type == "AlphaHead":
            head1 = get_head(head_config.head1, num_head_features)
            head2 = get_head(head_config.head2, num_head_features)
            return AlphaHead(head1, head2, alpha=head_config.alpha)
        else:
            assert False

        num_endpoints = len(datamodule.labels)
        if args.training.use_endpoint_embeddings:
            num_endpoints = datamodule.graph.num_features

        return cls(
            num_head_features,
            num_endpoints,
            incidence=incidence,
            dropout=head_config.dropout,
            gradient_checkpointing=args.training.gradient_checkpointing,
            **head_config.kwargs,
        )

    if "alpha_scheduler" in args.training and args.training:
        steps_per_epoch = math.ceil(len(datamodule.train_dataset) / args.datamodule.batch_size)
        max_epochs = args.trainer.max_epochs
        try:
            alpha_scheduler = hydra.utils.instantiate(
                args.training.alpha_scheduler,
                steps_per_epoch=steps_per_epoch,
                max_epochs=max_epochs,
            )
        except Exception:
            warnings.warn("Failed to instantiate AlphaScheduler, proceeding without it.")
            alpha_scheduler = None
    else:
        alpha_scheduler = None

    # TODO: fix
    if len(args.datamodule.covariates):
        num_covariates = len(args.datamodule.covariates) + 1
    else:
        num_covariates = 0

    incidence = datamodule.train_dataset.labels_events.mean(axis=0)
    incidence_mapping = {}
    fix_str = lambda s: s.replace(".", "-").replace("/", "-")
    for l, i in zip(datamodule.labels, np.array(incidence)[0]):
        l = fix_str(l)
        if i > 0.1:
            incidence_mapping[l] = ">1:10"
        elif (i <= 0.1) and (i > 0.01):
            incidence_mapping[l] = ">1:100"
        elif (i <= 0.01) and (i > 0.001):
            incidence_mapping[l] = ">1:1000"
        elif i <= 0.001:
            incidence_mapping[l] = "<1:1000"

    loss_weights = None
    if args.datamodule.use_loss_weights:
        loss_weights = 1 / np.log1p(1 / (incidence + 1e-8))

    losses = []
    metrics = []
    if args.datamodule.task == "binary":
        losses.append(
            EndpointClassificationLoss(
                FocalBCEWithLogitsLoss,
                {},
                scale=args.training.endpoint_loss_factor,
                use_exclusion_mask=args.training.exclusions_on_losses,
            )
        )
        metrics.append(torchmetrics.AUROC)
    elif args.datamodule.task == "tte":
        losses.append(
            EndpointTTELoss(
                CoxPHLoss,
                {},
                scale=args.training.endpoint_loss_factor,
                use_exclusion_mask=args.training.exclusions_on_losses,
                loss_weights=loss_weights,
            )
        )
        metrics.append(CIndex)

    modules = []
    if args.training.contrastive_loss_factor > 0:
        losses.append(EndpointContrastiveLoss(scale=args.training.contrastive_loss_factor))
        projector = torch.nn.Linear(args.head.kwargs["num_hidden"], args.model.num_outputs)
        modules.append(["projector", projector])

    endpoint_embedding_head = None
    if args.training.use_endpoint_embedding_head:
        endpoint_embedding_head = torch.nn.Linear(
            datamodule.graph.num_features, datamodule.graph.num_features
        )
        modules.append(["endpoint_embedding_head", endpoint_embedding_head])
    optimizer = None
    if args.training.optimizer == "Adam":
        optimizer = torch.optim.AdamW
    elif args.training.optimizer == "Shampoo":
        from torch_optimizer import Shampoo

        optimizer = Shampoo

    record_weights = torch.ones((len(datamodule.record_cols_input),), dtype=torch.float32)

    training_kwargs = dict(
        record_weights=record_weights,
        label_mapping=datamodule.label_mapping,
        incidence_mapping=incidence_mapping,
        exclusions_on_metrics=args.training.exclusions_on_metrics,
        losses=losses,
        metrics_list=metrics,
        alpha_scheduler=alpha_scheduler,
        optimizer=optimizer,
        node_dropout=args.training.node_dropout,
        normalize_node_embeddings=args.training.normalize_node_embeddings,
        optimizer_kwargs=args.training.optimizer_kwargs,
        binarize_records=args.training.binarize_records,
        use_endpoint_embeddings=args.training.use_endpoint_embeddings,
        test_time_augmentation_steps=args.training.test_time_augmentation_steps,
        use_lr_scheduler=args.training.use_lr_scheduler,
        record_weights_learnable=args.training.record_weights_learnable,
    )

    if args.model.model_type == "GNN":
        tags.append("gnn")

        gnn = HeteroGNN(
            datamodule.graph.num_features,
            args.model.num_hidden,
            args.model.num_outputs,
            args.model.num_blocks,
            metadata=datamodule.graph.metadata(),
            gradient_checkpointing=args.training.gradient_checkpointing,
            weight_norm=args.model.weight_norm,
            dropout=args.model.dropout,
        )

        num_head_features = gnn.num_outputs + num_covariates
        head = get_head(args.head, num_head_features)

        cls = RecordsShuffledGraphTraining if args.model.shuffled else RecordsGraphTraining

        modules.append(["graph_encoder", gnn])
        modules.append(["head", head])

        model = cls(
            modules=torch.nn.ModuleDict(modules),
            **training_kwargs,
        )
    elif args.model.model_type == "VICReg":
        tags.append("gnn")
        tags.append("vicreg")

        gnn = HeteroGNN(
            datamodule.graph.num_features,
            args.model.num_hidden,
            args.model.num_outputs,
            args.model.num_blocks,
            metadata=datamodule.graph.metadata(),
            gradient_checkpointing=args.training.gradient_checkpointing,
            weight_norm=args.model.weight_norm,
            dropout=args.model.dropout,
        )

        num_head_features = gnn.num_outputs + num_covariates
        head = get_head(args.head, num_head_features)

        modules.append(["graph_encoder", gnn])
        modules.append(["head", head])

        model = VICRegRecordsGraphTraining(
            vicreg_loss_scale=args.training.vicreg_loss_factor,
            modules=torch.nn.ModuleDict(modules),
            **training_kwargs,
        )
    elif args.model.model_type == "Identity":
        tags.append("identity")

        num_record_nodes = len(datamodule.record_node_indices)
        num_head_features = num_record_nodes + num_covariates
        head = get_head(args.head, num_head_features)

        modules.append(["head", head])

        model = RecordsIdentityTraining(modules=torch.nn.ModuleDict(modules), **training_kwargs)
    elif args.model.model_type == "GraphEmbeddings":
        tags.append("graph_embeddings")

        num_head_features = datamodule.num_features + num_covariates
        head = get_head(args.head, num_head_features)

        modules.append(["head", head])

        model = RecordsPretrainedEmbeddingsTraining(
            modules=torch.nn.ModuleDict(modules), **training_kwargs
        )
    elif args.model.model_type == "LearnedEmbeddings":
        tags.append("learned_embeddings")

        num_record_nodes = len(datamodule.record_node_indices)
        num_head_features = args.model.num_outputs + num_covariates
        head = get_head(args.head, num_head_features)
        embedder = torch.nn.Embedding(num_record_nodes, args.model.num_outputs)

        if args.model.freeze_embeddings:
            for param in embedder.parameters():
                param.requires_grad = False

        if args.datamodule.load_embeddings_path is not None:
            embedding_df = pd.read_feather(args.datamodule.load_embeddings_path)
            embedding_df.set_index("nodes", inplace=True)

            for i, c in enumerate(datamodule.record_cols_input):
                try:
                    e = embedding_df.loc[c].embeddings
                    embedder.weight.data[i] = torch.from_numpy(e)
                except KeyError as err:
                    print(f"Embedding missing for {err}")

        modules.append(["embedder", embedder])
        modules.append(["head", head])

        model = RecordsLearnedEmbeddingsTraining(
            modules=torch.nn.ModuleDict(modules), **training_kwargs
        )
    elif args.model.model_type == "Covariates":
        tags.append("covariates_baseline")

        num_head_features = num_covariates
        head = get_head(args.head, num_head_features)

        modules.append(["head", head])

        model = CovariatesOnlyTraining(modules=torch.nn.ModuleDict(modules), **training_kwargs)
    elif args.model.model_type == "SelfAttentionEmbeddings":
        tags.append("selfattention_embeddings")

        num_head_features = args.model.num_outputs + num_covariates
        head = get_head(args.head, num_head_features)
        embedding_dim = data.graph.x.shape[-1]

        modules.append(["head", head])
        # TODO: refactor -> add pma to modules

        model = RecordsSelfAttentionEmbeddingsTraining(
            embedding_dim,
            args.model.num_outputs,
            args.model.num_heads,
            modules=torch.nn.ModuleDict(modules),
            **training_kwargs,
        )
    else:
        assert False

    return datamodule, model, tags
