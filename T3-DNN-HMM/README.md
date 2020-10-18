# Task-3 : DNN-HMM training on TIMIT

## Preliminary
You must finish [Task-2](https://github.com/thuhcsi/DNN-HMM-Course/tree/main/T2-GMM-HMM) first to get a well-trained GMM-HMM model, which we will use for generating frame-level alignments.

If you have done Task-2, start your docker container built by Task-2 and navigate to timit folder:
```sh
# *** In host shell***
sudo docker start your_container_id
sudo docker attach your_container_id
# *** In docker shell***
cd /opt/kaldi/egs/timit/s5
. ./path.sh
```
Then you can check and view your GMM-HMM model by the following command:
```sh
# *** In docker shell***
# wirte model information to model.txt
gmm-copy --binary=false /opt/kaldi/egs/timit/s5/exp/tri3/final.mdl model.txt
```
Explanations of each part in `model.txt` can be found [here]()

Besides, you can also visualize your decoding graph (WFST) / MFCC features ...  :
```sh
# write graph information to HCLG.txt
fstprint /opt/kaldi/egs/timit/s5/exp/tri3/graph/HCLG.fst HCLG.txt
# write mfcc features to raw_mfcc_test.1.txt
copy-feats ark:/opt/kaldi/egs/timit/s5/mfcc/raw_mfcc_test.1.ark ark,t:raw_mfcc_test.1.txt
```

## Step-1
Run the DNN training:
```sh
# *** In docker shell***
./run.sh --stage 0
```
This script starts a full ASR experiment and performs data preparation, training, evaluation, forward, and decoding steps. A progress bar shows the evolution of all the aforementioned phases.


```sh
Total number of context-dependent phones: xxxx
============================================================================
                   Generate Alignments for DNN Training
============================================================================
......
Done align test set, alignment file: exp/data/ali_test.txt
============================================================================
                    Extract Features for DNN Training
============================================================================
......
Done apply cepstral mean and variance normalization (CMVN)
......
Done splice features with x left context and x right context
......
Dimention of transformed MFCC feature is xxx
============================================================================
                        DNN Training & Decoding
============================================================================
......
10/18/2020 13:57:59-***** Running DNN *****
10/18/2020 13:57:59-  total parameters: x.xxxxx M
10/18/2020 13:58:02-Train Epoch 0 | Step 1 | Loss 7.640 | Acc 0.08%
10/18/2020 13:58:05-Train Epoch 0 | Step 2 | Loss 7.308 | Acc 5.58%
10/18/2020 13:58:07-Train Epoch 0 | Step 3 | Loss 7.002 | Acc 7.41%
10/18/2020 13:58:10-Train Epoch 0 | Step 4 | Loss 6.820 | Acc 6.92%
10/18/2020 13:58:12-Train Epoch 0 | Step 5 | Loss 6.741 | Acc 5.55%
......
10/18/2020 13:15:49-Train Epoch 29 | Step 866 | Loss 2.352 | Acc 39.13%
10/18/2020 13:15:53-Train Epoch 29 | Step 867 | Loss 2.330 | Acc 39.20%
10/18/2020 13:15:56-Train Epoch 29 | Step 868 | Loss 2.315 | Acc 39.73%
10/18/2020 13:15:59-Train Epoch 29 | Step 869 | Loss 2.299 | Acc 40.08%
10/18/2020 13:16:01-Train Epoch 29 | Step 870 | Loss 2.232 | Acc 41.48%
10/18/2020 13:16:01-Evaluation DEV
10/18/2020 13:18:51-Epoch 29 DEV Loss 2.626 Acc 34.95% WER 23.9 | 400 15057 | 78.9 15.1 6.0 2.8 23.9 99.5 | -0.005 |
10/18/2020 13:18:51-Evaluation TEST
10/18/2020 13:20:16-Epoch 29 TEST Loss 2.654 Acc 34.19% WER 24.8 | 192 7215 | 78.0 15.3 6.7 2.8 24.8 99.5 | -0.032 |
```

The script `run.sh` progressively creates the following files in the output directory `$PWD/exp`:
* data/ali\_\*.ark: ......(WIP)
* data/ali\_\*.txt: ......(WIP)
* data/mfcc\_apply\_cmvn\_context\_\*.ark: ......(WIP)
* data/mfcc\_apply\_cmvn\_context\_\*.scp: ......(WIP)
* data/phone\_counts: ......(WIP)
* data/text\_\*: ......(WIP)

and the script `train.py` will save the trained model and write log to `$PWD/exp/$MODEL`.

The achieved WER(%) is 23.9%. The model we used is `DenseModel` in `model/dense.py`. This model is actually a toy model thus the performance is not really good. However, we will use it as our baseline since it is suitable to train on Desktop CPUs (i.e., time overhead is about 35min on Intel i5-8400).

**By correctly running the baseline scripts, you can get the basic score for this task.**


## Step-2
There are many ways to improve results, students who get better(lower) WERs will get extra points:
* Customize your own model in `model/mymodel.py` to get better WER, i.e., deeper model / CNN model / RNN model / Attention Model.
* Tricks for training neural networks, i.e., label smoothing / data augmentation. You need to implement those tricks by yourself if you want to use them.
* Try different parameters in `run.sh`, i.e., leftcontext=3 && rightcontext=3, this will provide useful information for frame classification. You can also increase those values to see if the result decreases and explain why.
* Try to extract different features, i.e., MFCC / FBANK / FMLLR / I-VECTOR. Generally speaking, the more features that are fused, the better results you can get.

NOTE: for feature extraction, we will show some examples here (WIP)
