from typing import Optional

import numpy as np
import scipy
import scipy.sparse
import torch


class RecordsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        records: scipy.sparse.csr_matrix,
        exclusions: scipy.sparse.csr_matrix,
        labels_events: scipy.sparse.csr_matrix,
        labels_times: scipy.sparse.csr_matrix,
        covariates: Optional[scipy.sparse.csr_matrix] = None,
        censorings: Optional[np.array] = None,
        eids: Optional[np.array] = None,
    ):
        self.records = records
        self.exclusions = exclusions
        self.covariates = covariates
        self.labels_events = labels_events
        self.labels_times = labels_times
        self.censorings = censorings
        self.eids = eids

    def __len__(self):
        return self.records.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        records = torch.Tensor(self.records[idx].todense())
        exclusions = torch.Tensor(self.exclusions[idx].todense())
        labels_events = torch.Tensor(self.labels_events[idx].todense())
        labels_times = torch.Tensor(self.labels_times[idx].todense())

        eids = torch.LongTensor([self.eids[idx]])

        covariates = None
        if self.covariates is not None:
            if not isinstance(idx, list):
                idx = [idx]
            covariates = torch.Tensor(self.covariates[idx])

        censorings = None
        if self.censorings is not None:
            if not isinstance(idx, list):
                idx = [idx]
            censorings = torch.Tensor(self.censorings[idx])

        data_tuple = (records, covariates)
        labels_tuple = (labels_events, labels_times, exclusions, censorings, eids)

        return data_tuple, labels_tuple
