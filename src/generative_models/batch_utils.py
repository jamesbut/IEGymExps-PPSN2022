# Utils for batching in torch

from torch.utils.data import DataLoader


def generate_batches(data, batch_size, shuffle):
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)
