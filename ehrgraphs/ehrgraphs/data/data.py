import datetime
import pathlib
import pickle
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch_geometric
import wandb
import yaml
import zstandard
from dask_ml import preprocessing
from tqdm.auto import tqdm


def load_wandb_artifact(
    identifier, run: wandb.sdk.wandb_run.Run = None, project: str = None, entity: str = None
):
    assert run is not None or (project is not None and entity is not None)

    if run is None:
        api = wandb.Api()
        artifact = api.artifact(f"{entity}/{project}/{identifier}")
    else:
        artifact = run.use_artifact(identifier)

    return artifact


def get_path_from_wandb(reference: str, data_root: Optional[str] = None):
    if data_root is not None:
        stub = pathlib.Path(reference.split("file://")[1]).name
        path = pathlib.Path(data_root, stub)
    else:
        path = pathlib.Path(reference.split("file://")[1])
    print(path)

    assert path.exists(), f"Path not found: {path}"
    return path


class WandBGraphData:
    def __init__(
        self,
        graph_name: str = "graph_full:latest",
        embedding_name: str = "full_prone_1024:latest",
        eid_dict_name: str = "eids:latest",
        individuals_name: str = "metadata_individuals:latest",
        covariates_name: str = "baseline_covariates:latest",
        records_data_name: str = "final_records_omop:latest",
        phecode_definitions_name: str = "phecode_definitions:latest",
        records_frequencies_name: str = "RecordFrequencies:latest",
        heterogeneous: bool = True,
        drop_shortcut_edges: bool = True,
        drop_individuals_without_records: bool = True,
        drop_individuals_without_gp: bool = False,
        min_record_counts: int = 0,
        data_root: Optional[str] = None,
        wandb_run: Optional[wandb.sdk.wandb_run.Run] = None,
        wandb_entity: Optional[str] = "cardiors",
        wandb_project: Optional[str] = "RecordGraphs",
        maximum_exit_date: Optional[datetime.datetime] = None,
    ):

        self.data_root = data_root
        self.graph_name = graph_name
        self.embedding_name = embedding_name

        self.eid_dict_name = eid_dict_name
        self.individuals_name = individuals_name
        self.covariates_name = covariates_name
        self.records_data_name = records_data_name
        self.phecode_definitions_name = phecode_definitions_name
        self.records_frequencies_name = records_frequencies_name

        self.heterogeneous = heterogeneous
        self.drop_shortcut_edges = drop_shortcut_edges
        self.drop_individuals_without_records = drop_individuals_without_records
        self.drop_individuals_without_gp = drop_individuals_without_gp
        self.min_record_counts = min_record_counts

        self.wandb_run = wandb_run
        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_project

        self.maximum_exit_date = maximum_exit_date

        self.phecode_definitions = self.load_phecode_definitions()
        self.records_frequencies = self.load_records_frequencies()
        embeddings = self.load_embeddings()

        self.individuals = self.load_individuals()
        records, valid_eids = self.load_records()
        self.eid_dict = self.load_eid_dict(valid_eids)

        self.covariates = self.load_covariates()

        self.record_encodings = self.get_record_encodings(records)
        self.eid_df, self.concept_df, self.record_df = self.normalize_records(
            records, self.record_encodings
        )

        graph = self.load_graph()
        self.graph = self.build_pyg_graph(graph, embeddings)

    def load_graph(self):
        graph_artifact = self._load_artifact(self.graph_name)
        entry = graph_artifact.manifest.entries[self.graph_name.split(":")[0]]
        graph_path = get_path_from_wandb(entry.ref, data_root=self.data_root)

        g = nx.readwrite.gpickle.read_gpickle(graph_path)

        if self.heterogeneous:
            g = self.preprocess_graph_heterogeneous(g)
        else:
            g = self.preprocess_graph_homogeneous(g)
        return g

    def load_phecode_definitions(self):
        phecode_artifact = self._load_artifact(self.phecode_definitions_name)

        phecodes_entries = phecode_artifact.manifest.entries
        phecodes_reference = phecodes_entries["phecode_definitions"]
        phecodes_path = get_path_from_wandb(phecodes_reference.ref, data_root=self.data_root)

        phecode_df = pd.read_feather(phecodes_path).set_index("index")
        return phecode_df

    def load_records_frequencies(self):
        records_artifact = self._load_artifact(self.records_frequencies_name)

        records_entries = records_artifact.manifest.entries
        records_reference = records_entries["RecordsMetadata"]
        records_path = get_path_from_wandb(records_reference.ref, data_root=self.data_root)

        records_df = pd.read_feather(records_path)
        return records_df

    def load_embeddings(self):
        embedding_artifact = self._load_artifact(self.embedding_name)

        embedding_entries = embedding_artifact.manifest.entries
        embedding_reference = embedding_entries["embeddings"]
        embedding_path = get_path_from_wandb(embedding_reference.ref, data_root=self.data_root)

        embedding_df = pd.read_feather(embedding_path).set_index("node")
        return embedding_df

    def load_eid_dict(self, valid_eids=None):
        artifact = self._load_artifact(self.eid_dict_name)
        entry = artifact.manifest.entries[self.eid_dict_name.split(":")[0]]
        path = get_path_from_wandb(entry.ref, data_root=self.data_root)
        eid_dict = yaml.load(open(path), Loader=yaml.CLoader)

        if valid_eids is not None:
            for partition, sets in eid_dict.items():
                for split, eids in sets.items():
                    eid_dict[partition][split] = np.array(
                        [eid for eid in eids if eid in valid_eids]
                    )

        return eid_dict

    def load_individuals(self):
        artifact = self._load_artifact(self.individuals_name)
        entry = artifact.manifest.entries[self.individuals_name.split(":")[0]]
        path = get_path_from_wandb(entry.ref, data_root=self.data_root)
        individuals_df = pd.read_feather(path)

        if self.maximum_exit_date is not None:
            individuals_df.exit_date = np.clip(
                individuals_df.exit_date, None, self.maximum_exit_date.date()
            )

        for col in tqdm(individuals_df.columns):
            if "date" in col:
                individuals_df[col] = individuals_df[col].astype("datetime64[D]")

        return individuals_df

    def load_covariates(self):
        artifact = self._load_artifact(self.covariates_name)
        entry = artifact.manifest.entries[self.covariates_name.split(":")[0]]
        path = get_path_from_wandb(entry.ref, data_root=self.data_root)
        covariates_df = pd.read_feather(path).set_index("eid")

        # drop all object cols for now
        # TODO fix encoding for object columns:
        covariates_df = covariates_df.apply(pd.to_numeric, errors="ignore")

        covariates_df = covariates_df[
            [c for c in covariates_df.columns.to_list() if "date" not in c]
        ]

        for c in covariates_df.columns:
            if covariates_df[c].dtype != "float64":
                covariates_df[c] = covariates_df[c].astype("category")

        return covariates_df

    def load_records(self):
        # TODO: sort by time asc
        artifact = self._load_artifact(self.records_data_name)
        entry = artifact.manifest.entries[self.records_data_name.split(":")[0]]
        path = get_path_from_wandb(entry.ref, data_root=self.data_root)
        records_df = pd.read_feather(path)

        if self.drop_individuals_without_gp:
            records_df["is_gp"] = records_df.origin.str.startswith("gp_")
            num_gp_records = records_df.groupby("eid").is_gp.sum()
            valid_eids = set(num_gp_records[num_gp_records > 0].index)
            records_df = records_df[records_df.eid.isin(valid_eids)]
        else:
            valid_eids = None

        # drop duplicate entries from different origins (hes/gp)
        # moved to datamodule due to hes/gp ablations
        # records_df = records_df.drop_duplicates(
        #    subset=[c for c in records_df.columns if c != "origin"]
        # ).copy()

        if self.min_record_counts > 0:
            concept_ids = records_df.concept_id.unique()
            frequent_omop_concepts = set(
                self.records_frequencies[
                    self.records_frequencies.n >= self.min_record_counts
                ].record
            )
            use_concepts = {
                l for l in concept_ids if l in frequent_omop_concepts or l.startswith("phecode")
            }
            # all_cause_death
            use_concepts.add("OMOP_4306655")

            records_df = records_df[records_df.concept_id.isin(use_concepts)]

        # remove records after min_record_counts filter to keep set of used records consitent
        if self.maximum_exit_date is not None:
            records_df.exit_date = np.clip(records_df.exit_date, None, self.maximum_exit_date)

        for col in tqdm(records_df.columns):
            if "date" in col:
                records_df[col] = records_df[col].astype("datetime64[D]")

        if self.drop_individuals_without_records:
            records_df = records_df[records_df.date.values <= records_df.exit_date.values]

        return records_df, valid_eids

    def preprocess_graph_heterogeneous(self, graph: nx.Graph):
        edge_types = []
        for u, v, data in graph.edges.data():
            edge_types.append(data["edge_type"])

        edge_codes, edge_types = pd.factorize(edge_types)

        node_types = []
        for n, data in graph.nodes.data():
            node_types.append(data["node_type"])

        node_codes, node_types = pd.factorize(node_types)

        preprocessed_graph = nx.DiGraph()
        preprocessed_graph.add_nodes_from(graph.nodes())

        preprocessed_graph.node_codes = node_codes
        preprocessed_graph.node_types = node_types

        omop_exclude_codes = []
        ot_exclude_codes = []
        if self.drop_shortcut_edges:
            omop_exclude_codes.append(list(edge_types).index("Subsumes"))
            omop_exclude_codes.append(list(edge_types).index("Is a"))

            # TODO: multiple indices?
            ot_exclude_codes.append(list(edge_types).index("Is descendant of"))
            ot_exclude_codes.append(list(edge_types).index("Is ancestor of"))

        for (u, v, w), c in zip(graph.edges.data("edge_weight"), edge_codes):
            assert w is not None

            # drop shortcut edges
            if c in omop_exclude_codes and w < 1.0:
                continue

            if c in ot_exclude_codes:
                continue

            preprocessed_graph.add_edge(u, v, edge_weight=w, edge_code=c)

        self.edge_types = edge_types

        return preprocessed_graph

    @staticmethod
    def preprocess_graph_homogeneous(graph: nx.Graph):
        simple_graph = nx.DiGraph()
        simple_graph.add_nodes_from(graph.nodes())

        for u, v, w in graph.edges.data("edge_weight"):
            assert w is not None
            simple_graph.add_edge(u, v, edge_weight=w)

        return simple_graph

    def build_pyg_graph(self, graph, embeddings):
        g = torch_geometric.utils.from_networkx(graph)

        assert not isinstance(g.x, torch.Tensor), "graph.x already populated"
        g.x = torch.from_numpy(np.stack(embeddings.embedding.values).astype(np.float32))
        g.node_ids = embeddings.index.values

        return g

    def _load_artifact(self, identifier: str):
        return load_wandb_artifact(
            identifier,
            self.wandb_run,
            self.wandb_project,
            self.wandb_entity,
        )

    def get_record_encodings(self, records):
        record_encodings = {}
        for col, dtype in [("eid", int), ("concept_id", str)]:
            if col == "eid":
                s = self.individuals["eid"]
            if col == "concept_id":
                s = records["concept_id"]
            s = s.drop_duplicates().dropna().astype(dtype).sort_values().reset_index(drop=True)
            cat_type = pd.api.types.CategoricalDtype(categories=s, ordered=False)
            le = preprocessing.LabelEncoder().fit(s.astype(cat_type))
            record_encodings[col] = (le, cat_type)
        return record_encodings

    def normalize_records(self, records, record_encodings):
        # normalize eids
        le_eids, cat_type_eid = record_encodings["eid"]
        self.individuals["eid_idx"] = le_eids.transform(
            self.individuals.eid.astype(int).astype(cat_type_eid)
        )
        eid_df = self.individuals[
            ["eid_idx", "eid", "birth_date", "recruitment_date", "death_date", "exit_date"]
        ].set_index("eid_idx")
        eid_df = eid_df[~eid_df.index.duplicated(keep="first")].copy()

        records["eid_idx"] = le_eids.transform(records.eid.astype(int).astype(cat_type_eid))

        # normalize concepts
        le_concepts, cat_type_concepts = record_encodings["concept_id"]
        records["concept_id_idx"] = le_concepts.transform(
            records["concept_id"].astype(str).astype(cat_type_concepts)
        )
        concept_df = records[
            ["concept_id_idx", "concept_id", "domain_id", "vocabulary", "origin"]
        ].set_index("concept_id_idx")
        concept_df = concept_df[~concept_df.index.duplicated(keep="first")].copy()

        # normalize records
        records_df = records[["eid_idx", "concept_id_idx", "date"]].copy()
        records_df.sort_values("date", ascending=True, inplace=True)

        return eid_df, concept_df, records_df


def get_or_load_wandbdataobj(
    data_root, identifier="WandBGraphData:latest", run=None, project=None, entity=None, **kwargs
):
    # log this as artifact:
    artifact = load_wandb_artifact(identifier, run=run, project=project, entity=entity)
    ref = artifact.manifest.entries["WandBGraphData"].ref
    stub = pathlib.Path(ref.split("file://")[1]).name
    artifact_path = pathlib.Path(data_root, stub)
    print(artifact_path)

    try:
        with open(artifact_path, "rb") as fh:
            dctx = zstandard.ZstdDecompressor()
            with dctx.stream_reader(fh) as decompressor:
                data = pickle.loads(decompressor.read())
    except Exception as err:
        print(err)
        data = WandBGraphData(
            data_root=data_root, wandb_run=run, wandb_entity=entity, wandb_project=project, **kwargs
        )
        # then pickle it:
        with open(artifact_path, "wb") as fh:
            cctx = zstandard.ZstdCompressor()
            with cctx.stream_writer(fh) as compressor:
                compressor.write(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))

    return data
