#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Xingchen Song @ 2020-10-13

import kaldiio
import numpy as np

from kaldiio import ReadHelper


def pad_list(xs, pad_value):
    """Perform padding for the list of numpy arrays.

    Args:
        xs (List): List of arrays [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding (default: 0 for features, -1 for alignments).

    Returns:
        pad (array): Padded array (B, Tmax, `*`).

    Examples:
        >>> x = [np.ones((4, 1)), np.ones((2, 1)), np.ones((1, 1))]
        >>> x
        [array([[1.], [1.], [1.],[1.]]), array([[1.], [1.]]), array([[1.]])]
        >>> pad_list(x, 0)
        array([[[1.], [1.], [1.],[1.]],
               [[1.], [1.], [0.],[0.]],
               [[1.], [0.], [0.],[0.]]]) # shape = (3, 4, 1)

    """
    n_batch = len(xs)
    max_len = max(x.shape[0] for x in xs)
    pad = np.zeros_like(xs[0], shape=(n_batch, max_len, xs[0].shape[-1]))
    pad.fill(pad_value)
    for i in range(n_batch):
        pad[i, : xs[i].shape[0]] = xs[i]

    return pad


def dataloader(alignments, features, batch_size, shuffle=False):
    """Loading alignments and features from kaldi files.

    Args:
        alignments (str): Absolute path of alignments file (Obtained from T2-GMM-HMM task).
        features (str): Absolute path of feats.scp file (Obtained from T2-GMM-HMM task).
        batch_size (int): batch size.
        shuffle (bool): shuffle the data if needed.

    Returns:
        id, data, target (generator): batched features and alignments with utt_id.

    """
    # first load aligments and save them in the dictionary
    align_reader = kaldiio.load_ark(alignments)
    align = {}
    for (utt_id, utt_align) in align_reader:
        align[utt_id] = utt_align

    # randomly read features and generate batch
    if shuffle:
        feats_reader = kaldiio.load_scp(features)
        data = None
    # sequentially read features and generate batch
    else:
        feats_reader = kaldiio.load_scp_sequential(features)
        data = (None, None)

    batch_idx = 0
    feats_buffer = []
    align_buffer = []
    id_buffer = []
    for data in feats_reader:
        if shuffle:
            utt_id = data
            utt_feat = feats_reader[utt_id]
        else:
            utt_id, utt_feat = data
        assert align[utt_id] is not None
        assert align[utt_id].shape[0] == utt_feat.shape[0]

        align_buffer.append(align[utt_id][:, np.newaxis])
        feats_buffer.append(utt_feat)
        id_buffer.append(utt_id)
        batch_idx += 1

        if batch_idx == batch_size:
            # bacth_size x max_feature_len x feature_dim
            data = pad_list(feats_buffer, pad_value=0.)
            # batch_szie x max_target_len
            target = pad_list(align_buffer, pad_value=-1)[:, :, 0]
            yield id_buffer, data, target

            batch_idx = 0
            feats_buffer = []
            align_buffer = []
            id_buffer = []

    # last batch
    if len(feats_buffer) > 0:
        # bacth_size x max_feature_len x feature_dim
        data = pad_list(feats_buffer, pad_value=0.)
        # batch_szie x max_target_len
        target = pad_list(align_buffer, pad_value=-1)[:, :, 0]
        yield id_buffer, data, target


if __name__ == "__main__":
    # just for test
    ali = 'exp/data/ali_dev.ark'
    feat = 'exp/data/mfcc_apply_cmvn_context_dev.scp'
    loader = dataloader(ali, feat, 1, True)
    for (uid, d, t) in loader:
        print(len(uid), d.shape, t.shape)
        print(uid[0], d[0, :, :], t[0, :])
        break
