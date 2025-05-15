import torch
from torch.utils.data import Dataset, Sampler, DataLoader
import numpy as np 
import logging
import pandas as pd
import copy

class DateGroupedBatchSampler(Sampler):
    """Sampler that groups data by date and returns entire groups as batches."""
    def __init__(self, data_source, shuffle=False, **kwargs):
        self.data_source = data_source
        self.shuffle = shuffle
        self.grouped_indices = self._group_indices_by_date()
        # self.batch_size = int(np.mean([len(group) for group in self.grouped_indices]))
        # self._dummy_batch_size = 1
    def _group_indices_by_date(self):
        indices = pd.Series(range(len(self.data_source)), index=self.data_source.get_index())
        grouped = indices.groupby(level='datetime').apply(list).values
        return grouped

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.grouped_indices)
        for group in self.grouped_indices:
            yield group

    def __len__(self):
        return len(self.grouped_indices)
    
    # @property
    # def batch_size(self):
    #     return self._dummy_batch_size

    
def collate_fn(batch):
    processed_batch = []
    for item in batch:
        # Check the percentage of NaNs

        item_df = pd.DataFrame(item)
        # Forward fill NaNs using pandas
        item_df.fillna(method='ffill', axis=0, inplace=True)
        item_df.fillna(method='bfill', axis=0, inplace=True)
        item_df.fillna(0, inplace=True)  # Fill remaining NaNs (if any at the start) with 0
        item_filled = item_df.to_numpy(dtype=np.float32)

        processed_batch.append(torch.tensor(item_filled, dtype=torch.float32))

    # # Ensure there is at least one valid item
    # if processed_batch:
    batch = torch.stack(processed_batch)
    B = batch.size(0)
    return batch.view(B, -1, batch.size(-1))  # reshape to (B, L, C)
    # else:
    #     # Return an empty tensor with the expected dimensions
    #     return torch.empty(0, 0, 0, dtype=torch.float32)
    

def init_data_loader(handler, shuffle, num_workers=4):
    date_grouped_sampler = DateGroupedBatchSampler(handler, shuffle)
    data_loader = DataLoader(handler,
                             batch_sampler=date_grouped_sampler,
                             collate_fn=collate_fn,
                             pin_memory=True,
                             persistent_workers=False,
                             num_workers=num_workers)
    return data_loader