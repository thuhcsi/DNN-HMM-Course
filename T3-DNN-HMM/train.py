#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Xingchen Song @ 2020-10-13

import os
import torch
import kaldiio
import logging
import argparse
import subprocess

import numpy as np

from dataloader import dataloader
from model.dense import DenseModel


def run_shell(cmd):
    """Running command lines to call kaldi functions

    Args:
        cmd (str): command lines

    Returns:
        output, err (str): runtime log
    """
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, shell=True)

    (output, err) = p.communicate()
    p.wait()

    return output, err


def cal_prior(prior_file):
    """Calculate prior probability $\mathcal{P}(HMMstate)$.
       Then we can simulate likelihood $\mathcal{P}(Observation|HMMstate)$
       with $\frac{\mathcal{P}(HMMstate|Observation)}{\mathcal{P}(HMMstate)}$

    Args:
        prior_file (str): path to prior_file

    Returns:
        prior (array): prior probability $\mathcal{P}(HMMstate)$

    """
    with open(prior_file) as f:
        row = next(f).strip().strip('[]').strip()
        counts = np.array([np.float32(v) for v in row.split()])
        prior = np.log(counts/np.sum(counts))

    return prior


def cal_loss_acc(pred, gold, ignore_id=-1):
    """Calculate cross entropy loss and accuracy.

    Args:
        pred (tensor): network output, batch_size x time x cdphones_num
        gold (tensor): ground truth, batch_size x time
        ignore_id (int): label id which should be ignored, default=-1

    Returns:
        loss (tensor): final loss
        num_correct / num_total (float): accuracy

    """
    cdphones_num = pred.size(2)
    log_prob = pred.contiguous().view(-1, cdphones_num)
    gold = gold.contiguous().view(-1).long()

    loss = torch.nn.functional.nll_loss(
        log_prob, gold, ignore_index=ignore_id, reduction='mean')

    pred = log_prob.argmax(dim=1)
    num_correct = torch.eq(pred, gold).sum().float().item()
    num_total = torch.ne(gold, -1).sum().float().item()

    return loss, num_correct / num_total


def evaluation(model, prior, loader, epoch, args, task='dev'):
    """forward model with PyTorch and run decoding with Kaldi

    Args:
        model (nn.Module): dnn model
        prior (array): prior probability $\mathcal{P}(HMMstate)$
        loader (dataloader): data generator
        epoch (int): current epoch
        args (arguments): parameters for decoding
        task (str): evaluation for dev set or test set

    Returns:
        output, err (str): runtime log
        total_loss / total_utt (float): average loss
        total_acc / total_utt (float): average accuracy

    """
    # forward model
    model.eval()
    likelihood_dict = {}
    total_loss = 0.
    total_acc = 0.
    total_utt = 0
    for (utt_id, utt_feat, utt_align) in loader:
        utt_feat = torch.from_numpy(utt_feat)
        utt_align = torch.from_numpy(utt_align)
        log_probs = model(utt_feat)
        loss, acc = cal_loss_acc(log_probs, utt_align, -1)
        total_loss += loss.item()
        total_acc += acc
        total_utt += len(utt_id)
        likelihood = log_probs[0, :, :].data.cpu().numpy().reshape(-1, args.cdphones_num) \
            - prior
        likelihood_dict[utt_id[0]] = likelihood

    # decoding is time consuming, especially at the early stages of training,
    # so we only do this for the last epoch.
    if args.epochs - epoch < 2:
        # write likelihood to ark-files
        if not os.path.exists(args.output_dir + '/decode_' + task):
            os.makedirs(args.output_dir + '/decode_' + task)
        ark_file = os.path.join(args.output_dir, 'decode_' + task,
                                'ep' + str(epoch) + task + '_likelihood.ark')
        kaldiio.save_ark(ark_file, likelihood_dict)
        # run kaldi for decoding
        ground_truth_dir = args.output_dir + '/../data/text_' + task
        cmd_decode = 'cd kaldi_decoding_script && ./decode_dnn.sh --kaldi-root ' + args.kaldi_root + ' ' + \
                     args.graph_dir + ' ' + ground_truth_dir + ' ' + \
                     args.gmmhmm + ' ' + ark_file + ' && cd ..'
        output, err = run_shell(cmd_decode)

    else:
        output = "".encode("utf-8")
        err = "".encode("utf-8")

    return output, err, total_loss / total_utt, total_acc / total_utt


def main(args):
    """Training loop and Evaluation

    Args:
        args (arguments): parameters for tarining and evaluation

    """

    # setting logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s-%(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    # setting seed
    logger.info(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # build model and optimizer
    model = DenseModel(args.feature_dim, args.cdphones_num)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    # make directory for saving model and writing log
    args.output_dir = os.path.join(
        os.getcwd(), args.output_dir, model.__class__.__name__)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger.info("***** Running DNN *****")
    logger.info('  total parameters: {0} M'.format(sum(param.numel()
                                                       for param in model.parameters()) / 1000000))

    global_step = 0
    for epoch in range(int(args.epochs)):

        # begin training
        model.train()
        loader = dataloader(args.align_file, args.feats_file,
                            args.batch_size, shuffle=True)
        for (utt_id, utt_feat, utt_align) in loader:
            utt_align = torch.from_numpy(utt_align)
            utt_feat = torch.from_numpy(utt_feat)
            log_probs = model(utt_feat)
            loss, acc = cal_loss_acc(log_probs, utt_align, ignore_id=-1)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            logger.info('Train Epoch {0} | Step {1} | Loss {2:.3f} | Acc {3:.2f}%'.format(
                epoch, global_step, loss.item(), acc * 100))

        # begin evaluation
        prior = cal_prior(args.prior_file)

        # dev set
        logger.info('Evaluation DEV')
        loader = dataloader(args.dev_align_file,
                            args.dev_feats_file, batch_size=1)
        log, err, dev_loss, dev_acc = evaluation(model, prior, loader, epoch,
                                                 args, 'dev')
        logger.info('Epoch {0} DEV Loss {1:.3f} Acc {2:.2f}% {3}'.format(
            epoch, dev_loss, dev_acc * 100, log.decode('utf-8')[1:].strip()))
        if err.decode('utf-8') != '':
            logger.info(err.decode('utf-8'))
            raise RuntimeError('Dev set Decoding Failed')

        # test set
        logger.info('Evaluation TEST')
        loader = dataloader(args.test_align_file,
                            args.test_feats_file, batch_size=1)
        log, err, test_loss, test_acc = evaluation(model, prior, loader, epoch,
                                                   args, 'test')
        logger.info('Epoch {0} TEST Loss {1:.3f} Acc {2:.2f}% {3}'.format(
            epoch, test_loss, test_acc * 100, log.decode('utf-8')[1:].strip()))
        if err.decode('utf-8') != '':
            logger.info(err.decode('utf-8'))
            raise RuntimeError('Test set Decoding Failed')

        # save final model
        if epoch == args.epochs - 1:
            model_file = os.path.join(
                args.output_dir, 'epoch'+str(epoch)+'.ckpt')
            torch.save(model.state_dict(), model_file)

    logger.info("***** Done DNN Training *****")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch DNN Model.')

    # training related
    parser.add_argument('--seed', type=int, default=3234,
                        help='random seed')
    parser.add_argument('--epochs', type=int, default=30,
                        help='total training epoch')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='number of sentences in one batch')
    parser.add_argument('--feature-dim', type=int, default=13,
                        help='MFCC/FBANK/MELSPECTROGRAM feature dimention')
    parser.add_argument('--cdphones-num', type=int, default=1904,
                        help='number of context-dependent phones(cdphones) \
                        run `hmm-info exp/tri3_ali/final.mdl` and check pdfs, \
                        cdphones-num is eaual to the number of pdfs')
    parser.add_argument('--output-dir', type=str, default='exp',
                        help='path to save the final model and write log')
    parser.add_argument('--align-file', type=str, default='exp/data/ali_train.ark',
                        help='path to train_alignment.ark')
    parser.add_argument('--feats-file', type=str, default='exp/data/mfcc_apply_cmvn_context_train.ark',
                        help='path to train_feats.ark')

    # decoding/evaluation related
    parser.add_argument('--prior-file', type=str, default='exp/data/phone_counts',
                        help='path to prior probability file')
    parser.add_argument('--dev-align-file', type=str, default='exp/data/ali_dev.ark',
                        help='path to dev_alignment.ark')
    parser.add_argument('--dev-feats-file', type=str, default='exp/data/mfcc_apply_cmvn_context_dev.ark',
                        help='path to dev_feats.ark')
    parser.add_argument('--test-align-file', type=str, default='exp/data/ali_test.ark',
                        help='path to test_alignment.ark')
    parser.add_argument('--test-feats-file', type=str, default='exp/data/mfcc_apply_cmvn_context_text.ark',
                        help='path to test_feats.ark')
    parser.add_argument('--graph-dir', type=str, default='/opt/kaldi/egs/timit/s5/exp/tri3/graph/',
                        help='directory containing graph(HCLG.fst) and lexicon(words.txt)')
    parser.add_argument('--gmmhmm', type=str, default='/opt/kaldi/egs/timit/s5/exp/tri3/final.mdl',
                        help='path to gmmhmm model')
    parser.add_argument('--kaldi-root', type=str, default='/opt/kaldi',
                        help='path to kaldi')

    args = parser.parse_args()
    main(args)
