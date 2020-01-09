import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.utils.data import TensorDataset, Dataset
from functools import reduce

import logging
logger = logging.getLogger(__name__)

class VisitSequenceWithLabelDataset(Dataset):
    def __init__(self, seqs, labels, num_features):
        """
        Args:
            seqs (list): list of patients (list) of visits (list) of codes (int) that contains visit sequences
            labels (list): list of labels (int)
            num_features (int): number of total features available
        """

        if len(seqs) != len(labels):
            raise ValueError("Seqs and Labels have different lengths {} {}"
                             .format(len(seqs), len(labels)))

        self.labels = labels

        # TODO: Complete this constructor to make self.seqs as a List of which each element represent visits of a patient
        # TODO: by Numpy matrix where i-th row represents i-th visit and j-th column represent the feature ID j.
        # TODO: You can use Sparse matrix type for memory efficiency if you want.
        self.seqs = [np.asarray(seq) for seq in seqs]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # returns will be wrapped as List of Tensor(s) by DataLoader
        return self.seqs[index], self.labels[index]


def visit_collate_fn(batch):
    """
    DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
    Thus, 'batch' is a list [(seq1, label1), (seq2, label2), ... , (seqN, labelN)]
    where N is minibatch size, seq is a (Sparse)FloatTensor, and label is a LongTensor

    :returns
        seqs (FloatTensor) - 3D of batch_size X max_length X num_features
        lengths (LongTensor) - 1D of batch_size
        labels (LongTensor) - 1D of batch_size
    """
    batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
    batch_size = len(batch)
    max_length = batch[0][0].shape[0]
    mapped = map(lambda x: x[0].shape[1], batch)
    num_features = reduce(lambda a, b: max(a, b), mapped)

    logger.debug('Tensor size: {} X {} X {}'.format(batch_size, max_length, num_features))
    lengths_tensor = torch.LongTensor(batch_size)
    labels_tensor = torch.LongTensor(batch_size)
    out = np.zeros((batch_size, max_length, num_features))

    for i, (seq, label) in enumerate(batch):
        out[i, :seq.shape[0], :seq.shape[1]] = seq
        lengths_tensor[i] = seq.shape[0]
        labels_tensor[i] = np.long(label)

    seqs_tensor = torch.FloatTensor(out)

    logger.debug('FloatTensor: {}'.format(seqs_tensor.shape))
    return (seqs_tensor, lengths_tensor), labels_tensor
